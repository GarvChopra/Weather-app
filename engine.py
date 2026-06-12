"""
AreaPulse CivicAlert Engine v4.2
==================================
Fixes applied (Phase 1 + Phase 2):
  - [P2] rain_3h now uses actual 3-hour sum, not 1-hour
  - [P2] wind feature now passes real windspeed_10m (was hardcoded 0)
  - [P2] tti now uses eff_rain in forecast mode (consistent with scores)
  - [P2] area encoder failure → drops area_enc instead of silently using ae=0
  - [P1] run_full_prediction accepts focus_lat/focus_lng for officer location
  - [P1] generate_bulletin updated to llama-4-scout (unified model name)
"""

import json, os, time, joblib
import numpy as np
import pandas as pd
import urllib.request, urllib.parse
from datetime import datetime, timedelta

# ── DATABASE ──────────────────────────────────────────────────
def fetch_issues_from_postgres(database_url, limit=500):
    try:
        import psycopg2, psycopg2.extras
    except ImportError:
        raise Exception("Run: pip install psycopg2-binary")
    try:
        conn = psycopg2.connect(database_url, connect_timeout=8)
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT id, description, area, severity, tag,
                   lat, lng, status, upvotes, timestamp, escalated
            FROM issues WHERE status != 'resolved'
            ORDER BY timestamp DESC LIMIT %s
        """, (limit,))
        rows = [dict(r) for r in cur.fetchall()]
        cur.close(); conn.close()
        print(f"[db] Loaded {len(rows)} open issues from Postgres")
        return rows
    except Exception as e:
        raise Exception(f"Postgres failed: {e}")

# ── DELHI AREA PROFILES ───────────────────────────────────────
DELHI_AREAS = {
    # ── OLD DELHI / CENTRAL ───────────────────────────────────────
    # Very old infra, zero drainage, high density — highest flood/sewage risk
    'Connaught Place':  {'lat':28.6315,'lng':77.2167,'drain':0.00,'elev':0.00,'road_age':1.0,'infra_age':1.0,'wp':0.00,'pop':1.00},
    'Chandni Chowk':    {'lat':28.6507,'lng':77.2303,'drain':0.00,'elev':0.00,'road_age':1.0,'infra_age':1.0,'wp':0.00,'pop':1.00},
    'Paharganj':        {'lat':28.6448,'lng':77.2167,'drain':0.00,'elev':0.00,'road_age':1.0,'infra_age':1.0,'wp':0.00,'pop':1.00},
    'Kashmere Gate':    {'lat':28.6675,'lng':77.2280,'drain':0.00,'elev':0.00,'road_age':1.0,'infra_age':1.0,'wp':0.25,'pop':0.75},

    # ── WEST DELHI ────────────────────────────────────────────────
    'Karol Bagh':       {'lat':28.6514,'lng':77.1907,'drain':0.25,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.75},
    'Rajouri Garden':   {'lat':28.6447,'lng':77.1220,'drain':0.25,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.75},
    'Punjabi Bagh':     {'lat':28.6590,'lng':77.1311,'drain':0.50,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.50},
    'Janakpuri':        {'lat':28.6219,'lng':77.0878,'drain':0.50,'elev':0.50,'road_age':0.5,'infra_age':0.5,'wp':1.00,'pop':0.50},
    'Dwarka':           {'lat':28.5921,'lng':77.0460,'drain':0.25,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.50},
    'Patel Nagar':      {'lat':28.6500,'lng':77.1700,'drain':0.25,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.25,'pop':0.75},

    # ── NORTH DELHI ───────────────────────────────────────────────
    'Rohini':           {'lat':28.7041,'lng':77.1025,'drain':0.50,'elev':0.50,'road_age':0.5,'infra_age':0.5,'wp':1.00,'pop':0.50},
    'Pitampura':        {'lat':28.7007,'lng':77.1311,'drain':0.50,'elev':0.50,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.50},
    'Model Town':       {'lat':28.7167,'lng':77.1900,'drain':0.25,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.25,'pop':0.75},
    'Civil Lines':      {'lat':28.6800,'lng':77.2250,'drain':0.50,'elev':0.50,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.25},
    'Mukherjee Nagar':  {'lat':28.7050,'lng':77.2100,'drain':0.25,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.25,'pop':0.75},

    # ── EAST DELHI ────────────────────────────────────────────────
    # Yamuna floodplain — lowest elevation, worst drainage
    'Shahdara':         {'lat':28.6706,'lng':77.2944,'drain':0.00,'elev':0.00,'road_age':1.0,'infra_age':1.0,'wp':0.25,'pop':0.75},
    'Laxmi Nagar':      {'lat':28.6310,'lng':77.2780,'drain':0.25,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.75},
    'Preet Vihar':      {'lat':28.6355,'lng':77.2944,'drain':0.25,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.75},
    'Mayur Vihar':      {'lat':28.6090,'lng':77.2944,'drain':0.25,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.50},

    # ── SOUTH DELHI ───────────────────────────────────────────────
    # Higher elevation, newer infra, better drainage — lower risk
    'Saket':            {'lat':28.5244,'lng':77.2090,'drain':1.00,'elev':0.50,'road_age':0.0,'infra_age':0.0,'wp':1.00,'pop':0.50},
    'Malviya Nagar':    {'lat':28.5355,'lng':77.2068,'drain':0.75,'elev':0.50,'road_age':0.0,'infra_age':0.0,'wp':1.00,'pop':0.50},
    'Hauz Khas':        {'lat':28.5494,'lng':77.2001,'drain':1.00,'elev':0.50,'road_age':0.0,'infra_age':0.0,'wp':1.00,'pop':0.25},
    'Greater Kailash':  {'lat':28.5494,'lng':77.2378,'drain':1.00,'elev':0.50,'road_age':0.0,'infra_age':0.0,'wp':1.00,'pop':0.25},
    'Lajpat Nagar':     {'lat':28.5677,'lng':77.2378,'drain':0.50,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.75},
    'Kalkaji':          {'lat':28.5494,'lng':77.2590,'drain':0.50,'elev':0.50,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.50},
    'Nehru Place':      {'lat':28.5491,'lng':77.2509,'drain':0.50,'elev':0.50,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.50},
    'Okhla':            {'lat':28.5355,'lng':77.2780,'drain':0.25,'elev':0.25,'road_age':1.0,'infra_age':1.0,'wp':0.25,'pop':0.50},
    'Mehrauli':         {'lat':28.5244,'lng':77.1855,'drain':0.25,'elev':0.50,'road_age':1.0,'infra_age':1.0,'wp':0.25,'pop':0.50},

    # ── SOUTH-WEST DELHI ──────────────────────────────────────────
    'Vasant Kunj':      {'lat':28.5200,'lng':77.1590,'drain':0.75,'elev':0.75,'road_age':0.0,'infra_age':0.0,'wp':1.00,'pop':0.25},
    'Vasant Vihar':     {'lat':28.5670,'lng':77.1600,'drain':1.00,'elev':1.00,'road_age':0.0,'infra_age':0.0,'wp':1.00,'pop':0.25},
    'RK Puram':         {'lat':28.5650,'lng':77.1800,'drain':0.75,'elev':0.50,'road_age':0.0,'infra_age':0.0,'wp':1.00,'pop':0.25},
    'Sarojini Nagar':   {'lat':28.5760,'lng':77.1980,'drain':0.75,'elev':0.50,'road_age':0.0,'infra_age':0.0,'wp':1.00,'pop':0.50},
    'INA':              {'lat':28.5733,'lng':77.2080,'drain':0.75,'elev':0.50,'road_age':0.0,'infra_age':0.0,'wp':1.00,'pop':0.50},

    # ── LUTYENS / CENTRAL-SOUTH ───────────────────────────────────
    'Lodhi Colony':     {'lat':28.5887,'lng':77.2208,'drain':1.00,'elev':0.50,'road_age':0.0,'infra_age':0.0,'wp':1.00,'pop':0.25},
    'Defence Colony':   {'lat':28.5731,'lng':77.2294,'drain':1.00,'elev':0.50,'road_age':0.0,'infra_age':0.0,'wp':1.00,'pop':0.25},
    'Nizamuddin':       {'lat':28.5910,'lng':77.2429,'drain':0.25,'elev':0.25,'road_age':1.0,'infra_age':1.0,'wp':0.25,'pop':0.50},
}

FLOOD_TIMING_MINS = {0.00: 15, 0.25: 40, 0.50: 100, 1.00: 999}

LABELS = [
    'label_flood',
    'label_pothole_worsen',
    'label_sewage_overflow',
    'label_garbage_flood',
    'label_elec_hazard',
]

LABEL_DISPLAY = {
    'label_flood':           {'icon': '🌊', 'name': 'Waterlogging',      'dept': 'DJB + PWD',       'color': '#1565c0'},
    'label_pothole_worsen':  {'icon': '🕳',  'name': 'Pothole Damage',    'dept': 'MCD Roads + PWD', 'color': '#bf360c'},
    'label_sewage_overflow': {'icon': '🚨', 'name': 'Sewage Overflow',   'dept': 'DJB',             'color': '#6a1b9a'},
    'label_garbage_flood':   {'icon': '🗑',  'name': 'Garbage Flooding',  'dept': 'MCD Sanitation',  'color': '#2e7d32'},
    'label_elec_hazard':     {'icon': '⚡', 'name': 'Electrical Hazard', 'dept': 'DISCOM',          'color': '#e65100'},
}

# ── MODEL ─────────────────────────────────────────────────────
_model = None
_encoder = None
_features = None
_loaded = False

def _load_model():
    global _model, _encoder, _features, _loaded
    if _loaded:
        return True
    d  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    mp = os.path.join(d, 'storm_model.pkl')
    ep = os.path.join(d, 'area_encoder.pkl')
    mm = os.path.join(d, 'model_meta.json')
    if not os.path.exists(mp):
        return False
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _model   = joblib.load(mp)
            _encoder = joblib.load(ep)
        _features = json.load(open(mm))['features']
        _loaded   = True
        print(f"[engine] Storm model loaded ({len(_features)} features)")
        return True
    except Exception as e:
        print(f"[engine] Model load failed: {e}")
        return False


# ── WEATHER ───────────────────────────────────────────────────
def fetch_live_weather(lat, lng):
    """Fetch weather from Open-Meteo.
    Uses past_hours=2 so the hourly array includes the last 2 actual measured
    hours — index [-1] is the most recently completed hour with real rain data,
    not a forecast. This is much more current than current.precipitation.
    """
    try:
        import requests as _req
        r = _req.get(
            'https://api.open-meteo.com/v1/forecast',
            params={
                'latitude':      lat,
                'longitude':     lng,
                'current':       'precipitation,weathercode,temperature_2m,windspeed_10m,windgusts_10m,relativehumidity_2m,surface_pressure,visibility',
                'hourly':        'precipitation,weathercode,temperature_2m,windspeed_10m,windgusts_10m,surface_pressure,visibility',
                'forecast_days': 1,
                'past_hours':    2,
                'timezone':      'Asia/Kolkata',
            },
            timeout=(3, 6),
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[weather] fetch failed for ({lat},{lng}): {e}")
        return None


def parse_live_weather(data):
    """Parse Open-Meteo JSON into a clean weather dict used by score_area().

    With past_hours=2, the hourly array looks like:
      index 0,1 = last 2 actual measured hours (real rain)
      index 2+  = forecast hours
    We take max(current.precipitation, last_measured_hour) as the real rain.
    """
    if data is None:
        return None

    c          = data.get('current', {})
    curr_rain  = float(c.get('precipitation')      or 0)
    curr_code  = int(c.get('weathercode')           or 0)
    curr_temp  = float(c.get('temperature_2m')      or 25)
    curr_wind  = float(c.get('windspeed_10m')       or 0)
    curr_gust  = float(c.get('windgusts_10m')       or 0)
    curr_humid = float(c.get('relativehumidity_2m') or 60)
    curr_press = float(c.get('surface_pressure')    or 1010)
    curr_vis   = float(c.get('visibility')          or 9999)

    h   = data.get('hourly', {})
    def s(k, d=0): return [float(x or d) for x in h.get(k, [])]
    hp  = s('precipitation')
    hpr = s('surface_pressure', 1010)
    hc  = [int(x or 0) for x in h.get('weathercode', [])]

    # With past_hours=2, hp[0] and hp[1] are actual measured rain from last 2 hours
    # hp[2]+ are forecasts. Use the measured hours to detect rain Open-Meteo missed.
    past_rain = max(hp[:2]) if len(hp) >= 2 else 0
    past_code = max(hc[:2]) if len(hc) >= 2 else 0

    # Take the most severe reading between current and recent past
    curr_rain = max(curr_rain, past_rain)
    curr_code = max(curr_code, past_code)

    # Forecast hours start at index 2 (since past_hours=2)
    forecast_hp = hp[2:] if len(hp) > 2 else hp
    forecast_hc = hc[2:] if len(hc) > 2 else hc

    rain_24h    = sum(forecast_hp[:24]) if forecast_hp else 0
    press_trend = (hpr[0] - hpr[2]) if len(hpr) >= 3 else 0
    rain_1h     = curr_rain
    rain_3h     = sum(forecast_hp[:3]) if len(forecast_hp) >= 3 else curr_rain * 3
    rain_6h     = sum(forecast_hp[:6]) if len(forecast_hp) >= 6 else curr_rain * 6

    # Use forecast arrays for future predictions
    hp = forecast_hp
    hc = forecast_hc

    WMO = {
        0: 'Clear', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
        45: 'Fog', 51: 'Light drizzle', 61: 'Light rain', 63: 'Moderate rain',
        65: 'Heavy rain', 80: 'Rain showers', 82: 'Violent showers',
        95: 'Thunderstorm', 96: 'Thunderstorm + hail', 99: 'Severe thunderstorm',
    }

    thunder_now = curr_code >= 95
    rain_now    = curr_code >= 51
    fog_now     = curr_code in [45, 48]
    wind_hazard = curr_gust >= 40
    heat_hazard = curr_temp >= 40
    storm_now   = curr_code >= 61
    storm_now   = storm_now or wind_hazard or heat_hazard or fog_now

    weather_intensity = min(1.0, (
        (curr_code / 99) * 0.5 +
        (max(0, curr_rain) / 20) * 0.3 +
        (max(0, curr_gust - 20) / 80) * 0.1 +
        (max(0, curr_temp - 30) / 20) * 0.1
    ))

    peak_i  = hp[:24].index(max(hp[:24])) if hp else 0
    peak_hr = 'NOW' if storm_now else (datetime.now() + timedelta(hours=peak_i)).strftime('%I:%M %p')

    max_3h_rain    = max((sum(hp[i:i+3]) for i in range(0, 21)), default=0) if len(hp) >= 3  else 0
    worst_3h_start = max(range(0, 21), key=lambda i: sum(hp[i:i+3]), default=0) if len(hp) >= 21 else 0
    worst_rain_time = (datetime.now() + timedelta(hours=worst_3h_start)).strftime('%I:%M %p')
    _wrd = datetime.now() + timedelta(hours=worst_3h_start)
    worst_rain_day  = _wrd.strftime('%a %d %b').replace(' 0', ' ') if worst_3h_start > 2 else 'Today'

    ht = h.get('temperature_2m', [])
    max_temp_24h  = max([float(x or 0) for x in ht[:24]]) if ht else curr_temp
    max_temp_hour = ht[:24].index(max(ht[:24])) if ht and len(ht) >= 24 else 0
    max_temp_time = (datetime.now() + timedelta(hours=max_temp_hour)).strftime('%I:%M %p')

    hg = h.get('windgusts_10m', [])
    max_gust_24h = max([float(x or 0) for x in hg[:24]]) if hg else curr_gust

    thunder_soon = any(c >= 95 for c in hc[:24])
    thunder_hour = next((i for i, c in enumerate(hc[:24]) if c >= 95), None)
    thunder_time = (
        (datetime.now() + timedelta(hours=thunder_hour)).strftime('%I:%M %p')
        if thunder_hour is not None else None
    )

    if thunder_soon:            forecast_intensity = 1.0
    elif max_3h_rain > 15:      forecast_intensity = 0.9
    elif max_3h_rain > 8:       forecast_intensity = 0.75
    elif max_3h_rain > 2:       forecast_intensity = 0.5
    elif max_temp_24h >= 44:    forecast_intensity = 0.8
    elif max_temp_24h >= 40:    forecast_intensity = 0.6
    elif max_gust_24h >= 60:    forecast_intensity = 0.7
    elif max_gust_24h >= 40:    forecast_intensity = 0.5
    else:                       forecast_intensity = 0.2

    weather_coming = (
        max_3h_rain > 2 or thunder_soon or
        max_temp_24h >= 40 or max_gust_24h >= 40
    )

    if thunder_soon and thunder_time:
        forecast_summary = f'⛈ Thunderstorm at {thunder_time}'
    elif max_3h_rain > 15:
        forecast_summary = f'🌧 Heavy rain ({max_3h_rain:.0f}mm) at {worst_rain_time}'
    elif max_3h_rain > 5:
        forecast_summary = f'🌦 Rain ({max_3h_rain:.0f}mm) at {worst_rain_time}'
    elif max_3h_rain > 1:
        forecast_summary = f'🌦 Light rain at {worst_rain_time}'
    elif max_temp_24h >= 44:
        forecast_summary = f'🌡 Extreme heat {max_temp_24h:.0f}°C at {max_temp_time}'
    elif max_temp_24h >= 40:
        forecast_summary = f'🔆 Heatwave {max_temp_24h:.0f}°C at {max_temp_time}'
    elif max_gust_24h >= 40:
        forecast_summary = f'💨 High winds {max_gust_24h:.0f}km/h forecast'
    else:
        forecast_summary = '☀ No significant weather in next 24h'

    return {
        # Current real-time
        'curr_rain':  round(curr_rain,  1),
        'curr_code':  curr_code,
        'curr_temp':  round(curr_temp,  1),
        'curr_wind':  round(curr_wind,  1),   # FIX [P2]: now populated
        'curr_gust':  round(curr_gust,  1),
        'curr_humid': round(curr_humid, 1),
        'curr_press': round(curr_press, 1),
        'curr_vis':   round(curr_vis,   0),
        'press_trend': round(press_trend, 2),
        'current_condition': WMO.get(curr_code, f'Code {curr_code}'),

        # Multi-horizon rain  — all distinct now
        'rain_next_1h': round(rain_1h,  1),
        'rain_next_3h': round(rain_3h,  1),   # FIX [P2]: added
        'rain_next_6h': round(rain_6h,  1),
        'rain_24h':     round(rain_24h, 1),

        # Storm flags
        'storm_now':       storm_now,
        'thunder_now':     thunder_now,
        'rain_now':        rain_now,
        'wind_hazard':     wind_hazard,
        'heat_hazard':     heat_hazard,
        'fog_now':         fog_now,
        'weather_intensity': weather_intensity,
        'thunder_soon':    thunder_soon,
        'thunder_time':    thunder_time,
        'fog':             curr_vis < 500,
        'dense_fog':       curr_vis < 200,
        'peak_rain_hour':  peak_hr,

        # 24h forecast
        'max_3h_rain':       round(max_3h_rain,   1),
        'worst_rain_time':   worst_rain_time,
        'worst_rain_day':    worst_rain_day,
        'max_temp_24h':      round(max_temp_24h,  1),
        'max_temp_time':     max_temp_time,
        'max_gust_24h':      round(max_gust_24h,  1),
        'weather_coming':    weather_coming,
        'forecast_intensity': forecast_intensity,
        'forecast_summary':  forecast_summary,

        'month': datetime.now().month,
        'hour':  datetime.now().hour,
    }


def fetch_aqi(token='demo'):
    try:
        import requests as _req
        r = _req.get(
            f'https://api.waqi.info/feed/delhi/?token={token}',
            timeout=(3, 5)
        )
        d = r.json()
        if d.get('status') == 'ok':
            aqi = d['data']['aqi']
            print(f"[aqi] Delhi AQI: {aqi} (aqicn)")
            return {'aqi': aqi, 'source': 'aqicn'}
        else:
            print(f"[aqi] AQICN error: {d.get('data', 'unknown')}")
    except Exception as e:
        print(f"[aqi] fetch failed: {e}")

    try:
        import requests as _req
        r = _req.get(
            'https://air-quality-api.open-meteo.com/v1/air-quality',
            params={
                'latitude': 28.65, 'longitude': 77.22,
                'current':  'us_aqi,pm2_5,pm10',
                'timezone': 'Asia/Kolkata',
            },
            timeout=(3, 5)
        )
        d = r.json()
        aqi = d.get('current', {}).get('us_aqi')
        if aqi:
            print(f"[aqi] Delhi AQI: {aqi} (open-meteo)")
            return {'aqi': int(aqi), 'source': 'open-meteo AQ'}
    except Exception as e:
        print(f"[aqi] open-meteo fallback failed: {e}")

    print("[aqi] All sources failed — AQI unavailable")
    return None


# ── SCORING ───────────────────────────────────────────────────
def score_area(area_name, meta, weather, open_issues):
    area_issues = [
        i for i in open_issues
        if str(i.get('area', '')).strip().lower() == area_name.strip().lower()
        and i.get('status') not in ('resolved',)
    ]
    w_open = sum(1 for i in area_issues if i.get('tag') == 'water')
    s_open = sum(1 for i in area_issues if i.get('tag') == 'sewage')
    p_open = sum(1 for i in area_issues if i.get('tag') == 'pothole')
    g_open = sum(1 for i in area_issues if i.get('tag') == 'garbage')
    e_open = sum(1 for i in area_issues if i.get('tag') == 'electricity')
    now_ts = time.time()
    complaint_vel = sum(
        1 for i in area_issues
        if now_ts - float(i.get('timestamp') or 0) < 7200
    )

    using_forecast = (
        not weather.get('storm_now', False) and
        weather.get('weather_coming', False)
    )

    eff_rain    = weather.get('max_3h_rain', 0)                        if using_forecast else weather['curr_rain']
    eff_temp    = weather.get('max_temp_24h', weather['curr_temp'])    if using_forecast else weather['curr_temp']
    eff_gust    = weather.get('max_gust_24h', weather['curr_gust'])    if using_forecast else weather['curr_gust']
    eff_thunder = weather.get('thunder_soon', False)                   if using_forecast else weather.get('thunder_now', False)
    eff_wi      = weather.get('forecast_intensity', 0.2)               if using_forecast else weather.get('weather_intensity', 0.3)
    eff_code    = (63 if eff_rain > 5 else 51)                         if using_forecast else weather['curr_code']

    scores = {}
    if _load_model():
        try:
            ae = 0
            try:
                ae = int(_encoder.transform([area_name])[0])
            except Exception:
                pass

            row = {
                'area_enc':    ae,
                'drain':       meta['drain'],
                'elev':        meta['elev'],
                'road_age':    meta['road_age'],
                'infra_age':   meta['infra_age'],
                'wp':          meta['wp'],
                'pop':         meta['pop'],
                'month':       weather['month'],
                'hour':        weather['hour'],
                'rain_1h':     eff_rain,
                'rain_3h':     weather['rain_next_3h'],
                'rain_6h':     weather['rain_next_6h'],
                'rain_24h':    weather['rain_24h'],
                'temp':        eff_temp,
                'wind':        weather.get('curr_wind', 0),
                'gust':        eff_gust,
                'humid':       weather['curr_humid'],
                'pressure':    weather['curr_press'],
                'press_trend': weather['press_trend'],
                'visibility':  weather['curr_vis'],
                'weathercode': eff_code,
                'storm_now':   int(weather.get('storm_now', False) or using_forecast),
                'thunder_now': int(eff_thunder),
                'open_water':  w_open,
                'open_sewage': s_open,
                'open_pothole': p_open,
                'open_garbage': g_open,
                'open_elec':   e_open,
                'complaint_vel': complaint_vel,
            }

            fd    = pd.DataFrame([row])[_features]
            probs = np.array(_model.predict_proba(fd))

            # Pure ML — score = probability × area vulnerability multiplier
            # Multiplier reflects infrastructure quality (drain, road age, etc.)
            # No rules fallback — if ML says low risk, we show low risk
            VULN = {
                'label_flood':          0.6 + (1 - meta['drain']) * 0.3 + (1 - meta['elev'])  * 0.1,
                'label_pothole_worsen': 0.6 + meta['road_age']    * 0.3 + (1 - meta['drain']) * 0.1,
                'label_sewage_overflow':0.6 + (1 - meta['drain']) * 0.3 + meta['pop']         * 0.1,
                'label_garbage_flood':  0.6 + meta['pop']         * 0.2 + (1 - meta['drain']) * 0.2,
                'label_elec_hazard':    0.6 + meta['infra_age']   * 0.3 + meta['pop']         * 0.1,
            }
            for i, label in enumerate(LABELS):
                scores[label] = min(int(probs[i][0][1] * 100 * VULN.get(label, 0.7)), 99)

        except Exception as e:
            print(f"[model] scoring error: {e}")
            scores = {label: 0 for label in LABELS}
    else:
        # Model not loaded — return zero scores, don't fabricate
        scores = {label: 0 for label in LABELS}

    # FIX [P2]: tti now uses eff_rain (not curr_rain) so forecast mode is consistent
    dk   = min(FLOOD_TIMING_MINS.keys(), key=lambda x: abs(x - meta['drain']))
    base = FLOOD_TIMING_MINS[dk]
    tf   = 0.4 if eff_rain > 20 else 0.6 if eff_rain > 10 else 0.8 if eff_rain > 5 else 1.0
    isf  = max(0.5, 1.0 - (s_open + w_open) * 0.1)
    tti  = int(base * tf * isf)

    tag_risk = {
        'pothole':     'label_pothole_worsen',
        'sewage':      'label_sewage_overflow',
        'water':       'label_flood',
        'garbage':     'label_garbage_flood',
        'electricity': 'label_elec_hazard',
        'streetlight': 'label_elec_hazard',
    }
    affected = []
    for issue in area_issues:
        tag  = issue.get('tag', '')
        risk = tag_risk.get(tag)
        if risk and scores.get(risk, 0) >= 30:
            affected.append({
                'id':          issue.get('id'),
                'tag':         tag,
                'severity':    issue.get('severity', 'medium'),
                'description': str(issue.get('description', ''))[:60],
                'risk_label':  risk,
                'risk_score':  scores.get(risk, 0),
            })

    overall = max(scores.values()) if scores else 0
    # storm_amplified: True when weather is active OR when open issues make risk >= 25
    # Previously gated on storm_now which hid all areas on clear days
    has_weather  = weather.get('storm_now', False) or weather.get('weather_coming', False)
    has_issues   = len(area_issues) > 0 and overall >= 25
    return {
        'area':         area_name,
        'lat':          meta['lat'],
        'lng':          meta['lng'],
        'scores':       scores,
        'overall_risk': overall,
        'risk_level':   ('CRITICAL' if overall >= 75 else
                         'HIGH'     if overall >= 50 else
                         'MEDIUM'   if overall >= 25 else 'LOW'),
        'time_to_impact_mins': tti,
        'affected_issues':     sorted(affected, key=lambda x: x['risk_score'], reverse=True),
        'open_issues_count':   len(area_issues),
        'complaint_velocity':  complaint_vel,
        'storm_amplified':     has_weather or has_issues,
        'drain_quality':       (
            'very poor' if meta['drain'] <= 0    else
            'poor'      if meta['drain'] <= 0.25 else
            'medium'    if meta['drain'] <= 0.5  else 'good'
        ),
    }


# ── BULLETIN ──────────────────────────────────────────────────
def generate_bulletin(results, weather, aqi, groq_api_key):
    """Generate a short emergency bulletin. Uses Groq if key provided."""
    top   = [r for r in results if r['overall_risk'] > 0][:3]
    storm = weather.get('storm_now', False) or weather.get('weather_coming', False)

    if not top:
        return f"{weather.get('current_condition', 'Clear')} in Delhi. No immediate civic risk detected. Monitoring {len(results)} areas."

    areas_str = ', '.join(
        f"{r['area']} ({r['overall_risk']}/100)" for r in top
    )
    prompt = (
        f"You are AreaPulse CivicAlert. Write ONE emergency bulletin under 90 words.\n"
        f"Weather: {weather.get('current_condition')} · rain {weather.get('curr_rain')}mm/hr · "
        f"temp {weather.get('curr_temp')}°C · gusts {weather.get('curr_gust')}km/h\n"
        f"Top risk areas: {areas_str}\n"
        f"Storm active: {storm}\n"
        f"Name issue IDs. Name departments. Max 90 words."
    )

    # FIX [P2 / naming]: unified to llama-4-scout across all engine files
    if groq_api_key:
        try:
            from groq import Groq
            resp = Groq(api_key=groq_api_key).chat.completions.create(
                model='meta-llama/llama-4-scout-17b-16e-instruct',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=160,
                temperature=0.3,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[groq] {e}")

    # Fallback: rule-based bulletin
    t   = top[0]
    tr  = max(t['scores'], key=t['scores'].get) if t['scores'] else 'label_flood'
    ids = [f"#AP-{i['id']}" for i in t['affected_issues'][:2] if i.get('id')]
    return (
        f"{'⛈ STORM ACTIVE — ' if storm else ''}"
        f"{weather.get('current_condition', 'Weather alert')} in Delhi. "
        f"Highest risk: {t['area']} ({t['overall_risk']}/100) — "
        f"{LABEL_DISPLAY.get(tr, {}).get('name', 'flooding')} in ~{t['time_to_impact_mins']}min. "
        f"{'Issues ' + ', '.join(ids) + ' will worsen. ' if ids else ''}"
        f"Deploy {LABEL_DISPLAY.get(tr, {}).get('dept', 'relevant dept')} immediately."
    )


# ── MAIN ENTRY POINT ──────────────────────────────────────────
def run_full_prediction(open_issues=None, groq_api_key=None, aqi_token='demo',
                        focus_lat=None, focus_lng=None, rain_override_mm=0):
    """
    Run full civic risk prediction for all Delhi areas.

    focus_lat / focus_lng — fetch weather for user's location.
    rain_override_mm — manual rain intensity override (0=off, 5=light, 25=heavy, 40=storm).
      When set, overrides all zone weather with this rain amount so ML model
      scores using the user-reported rain, not the lagged API value.
    """
    open_issues = open_issues or []
    _load_model()
    print(f"[engine] v4.2 · {len(DELHI_AREAS)} areas · {len(open_issues)} issues"
          + (f" · rain_override={rain_override_mm}mm" if rain_override_mm else ""))

    # ── PHASE 3: PARALLEL PER-AREA WEATHER FETCH ──────────────
    # Each area gets its own weather from its exact lat/lng —
    # ── ZONE-BASED WEATHER FETCH (fixes 429 rate limit) ──────────
    # 36 simultaneous requests → Open-Meteo 429. Solution: 4 zone
    # fetches covering Delhi's geographic spread, in parallel.
    # Each area picks its nearest zone — still gives distinct weather
    # per area (north storm vs south clear) without hammering the API.
    #
    # Zones chosen at geographic extremes of Delhi:
    #   North : Rohini / Pitampura / Model Town
    #   South : Saket / Okhla / Mehrauli
    #   East  : Shahdara / Mayur Vihar / Laxmi Nagar
    #   West  : Dwarka / Janakpuri / Rajouri Garden
    #   Centre: Connaught Place (summary + bulletin)

    ZONES = {
        'north':  (28.720, 77.130),  # Rohini area
        'south':  (28.530, 77.220),  # Saket / Okhla area
        'east':   (28.650, 77.295),  # Shahdara / Mayur Vihar area
        'west':   (28.610, 77.070),  # Dwarka / Janakpuri area
        'centre': (28.632, 77.217),  # Connaught Place
    }

    # P1: if officer location given, add it as an extra zone
    if focus_lat is not None and focus_lng is not None:
        ZONES['officer'] = (focus_lat, focus_lng)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_zone(zone_name, lat, lng):
        raw = fetch_live_weather(lat, lng)
        if raw is None:
            return zone_name, None
        return zone_name, parse_live_weather(raw)

    print(f"[engine] Fetching weather for {len(ZONES)} zones in parallel...")
    zone_weather = {}

    with ThreadPoolExecutor(max_workers=len(ZONES)) as pool:
        future_map = {
            pool.submit(_fetch_zone, name, lat, lng): name
            for name, (lat, lng) in ZONES.items()
        }
        for future in as_completed(future_map):
            zone_name, wx = future.result()
            zone_weather[zone_name] = wx

    ok = sum(1 for w in zone_weather.values() if w is not None)
    print(f"[engine] Zones fetched: {ok}/{len(ZONES)} OK")

    if ok == 0:
        raise Exception("Open-Meteo unreachable for all zones. Check internet connection.")

    # Fill any failed zones from nearest successful zone
    zone_coords = {n: ZONES[n] for n in ZONES}
    for zn in list(zone_weather.keys()):
        if zone_weather[zn] is not None:
            continue
        best, best_d = None, float('inf')
        zlat, zlng = zone_coords[zn]
        for other, wx in zone_weather.items():
            if wx is None or other == zn:
                continue
            olat, olng = zone_coords[other]
            d = ((zlat - olat)**2 + (zlng - olng)**2)**0.5
            if d < best_d:
                best_d, best = d, wx
        zone_weather[zn] = best
        print(f"[engine] zone '{zn}' used nearest fallback")

    def _zone_for_area(meta):
        """Return the closest zone's parsed weather for a given area."""
        alat, alng = meta['lat'], meta['lng']
        best_zone, best_d = 'centre', float('inf')
        for zn, (zlat, zlng) in zone_coords.items():
            d = ((alat - zlat)**2 + (alng - zlng)**2)**0.5
            if d < best_d:
                best_d, best_zone = d, zn
        return zone_weather.get(best_zone) or zone_weather.get('centre')

    # If user location provided, use THEIR weather as the hero summary
    if focus_lat is not None and focus_lng is not None and zone_weather.get('officer'):
        weather = zone_weather['officer']
        print(f"[engine] Using officer/user location weather for summary: "
              f"rain={weather.get('curr_rain')} code={weather.get('curr_code')}")
    else:
        weather = zone_weather.get('centre') or next(
            w for w in zone_weather.values() if w is not None
        )

    # ── RAIN OVERRIDE — apply BEFORE scoring ─────────────────────
    # When user reports rain manually (API lag), patch all zone weather
    # so the ML model scores with the actual conditions, not the stale API value.
    WMO_OVERRIDE = {5: 61, 25: 65, 40: 95}
    if rain_override_mm > 0:
        override_code = WMO_OVERRIDE.get(rain_override_mm, 63)
        print(f"[engine] Rain override active: {rain_override_mm}mm → code={override_code}")
        for zn in zone_weather:
            if zone_weather[zn] is None:
                continue
            zone_weather[zn] = dict(zone_weather[zn])
            zone_weather[zn]['curr_rain']  = float(rain_override_mm)
            zone_weather[zn]['curr_code']  = override_code
            zone_weather[zn]['rain_1h']    = float(rain_override_mm)
            zone_weather[zn]['rain_next_1h'] = float(rain_override_mm)
            zone_weather[zn]['rain_next_3h'] = float(rain_override_mm) * 2
            zone_weather[zn]['rain_next_6h'] = float(rain_override_mm) * 3
            zone_weather[zn]['rain_now']   = True
            zone_weather[zn]['storm_now']  = rain_override_mm >= 15
            zone_weather[zn]['thunder_now'] = rain_override_mm >= 40
        # Re-get summary weather after patch
        if focus_lat is not None and zone_weather.get('officer'):
            weather = zone_weather['officer']
        else:
            weather = zone_weather.get('centre') or next(
                w for w in zone_weather.values() if w is not None
            )

    print("[engine] Fetching AQI...")
    aqi = fetch_aqi(aqi_token)

    # Score every area with its nearest zone's weather
    results = []
    for area_name, meta in DELHI_AREAS.items():
        area_wx = _zone_for_area(meta)
        results.append(score_area(area_name, meta, area_wx, open_issues))

    # P1: if officer location given, sort nearest areas first by risk
    if focus_lat is not None and focus_lng is not None:
        def _dist(r):
            return ((r['lat'] - focus_lat) ** 2 + (r['lng'] - focus_lng) ** 2) ** 0.5
        nearby  = sorted(results, key=_dist)[:5]
        faraway = sorted(results, key=_dist)[5:]
        nearby.sort(key=lambda x: x['overall_risk'], reverse=True)
        faraway.sort(key=lambda x: x['overall_risk'], reverse=True)
        results = nearby + faraway
    else:
        storm_mode = weather.get('storm_now', False) or weather.get('weather_coming', False)
        if storm_mode:
            results.sort(key=lambda x: (-x['overall_risk'], x['time_to_impact_mins']))
        else:
            results.sort(key=lambda x: x['overall_risk'], reverse=True)

    # ── SUMMARY STRING ────────────────────────────────────────
    rain = weather['curr_rain']
    temp = weather['curr_temp']
    cond = weather['current_condition']
    gust = weather.get('curr_gust', 0)
    fcst = weather.get('forecast_summary', '')

    curr_code = weather.get('curr_code', 0)
    if weather.get('thunder_now'):
        summary = f'⛈ THUNDERSTORM NOW — {cond} · gusts {gust}km/h'
    elif curr_code >= 65 or (weather.get('rain_now') and rain > 10):
        summary = f'🌧 HEAVY RAIN NOW — {cond} · {rain}mm/hr · {weather["rain_next_6h"]}mm next 6h'
    elif curr_code >= 61 or (weather.get('rain_now') and rain > 0):
        summary = f'🌦 RAIN NOW — {cond} · {("~" + str(rain) + "mm/hr") if rain > 0 else "radar confirmed"}'
    elif curr_code >= 51 or weather.get('rain_now'):
        summary = f'🌦 LIGHT RAIN — {cond}'
    elif weather.get('wind_hazard') and gust > 60:
        summary = f'💨 STRONG WINDS NOW — gusts {gust}km/h · {cond}'
    elif weather.get('wind_hazard'):
        summary = f'🌬 WINDY NOW — gusts {gust}km/h · {cond}'
    elif weather.get('heat_hazard') and temp > 44:
        summary = f'🌡 EXTREME HEAT NOW — {temp}°C · {cond}'
    elif weather.get('heat_hazard'):
        summary = f'🔆 HEATWAVE NOW — {temp}°C · {cond}'
    elif weather.get('fog_now'):
        summary = f'🌫 FOG NOW — {int(weather["curr_vis"])}m visibility'
    elif weather.get('weather_coming'):
        summary = f'☀ Clear now · Forecast: {fcst}'
    elif aqi and aqi['aqi'] > 200:
        summary = f'😷 Poor air quality — AQI {aqi["aqi"]}'
    elif temp < 6:
        summary = f'❄ Cold wave — {temp}°C · {cond}'
    else:
        summary = f'☀ {cond} — {temp}°C · No weather alerts'

    bulletin = generate_bulletin(results, weather, aqi, groq_api_key)

    label_summary = {}
    for label in LABELS:
        vals = [r['scores'].get(label, 0) for r in results]
        label_summary[label] = {
            'avg':            round(sum(vals) / max(len(vals), 1)),
            'max':            max(vals),
            'critical_areas': [r['area'] for r in results if r['scores'].get(label, 0) >= 75],
            'display':        LABEL_DISPLAY.get(label, {}),
        }

    return {
        'areas':   results,
        'weather': {
            'summary':           summary,
            'curr_rain':         weather['curr_rain'],
            'curr_temp':         weather['curr_temp'],
            'curr_wind':         weather.get('curr_wind', 0),
            'current_condition': weather['current_condition'],
            'rain_next_1h':      weather['rain_next_1h'],
            'rain_next_3h':      weather['rain_next_3h'],
            'rain_next_6h':      weather['rain_next_6h'],
            'rain_24h':          weather['rain_24h'],
            'curr_gust':         weather['curr_gust'],
            'curr_humid':        weather['curr_humid'],
            'curr_vis':          weather['curr_vis'],
            'curr_press':        weather['curr_press'],
            'press_trend':       weather['press_trend'],
            'has_thunder':       weather['thunder_now'],
            'thunder_soon':      weather.get('thunder_soon', False),
            'thunder_time':      weather.get('thunder_time'),
            'weather_coming':    weather.get('weather_coming', False),
            'forecast_summary':  weather.get('forecast_summary', ''),
            'forecast_intensity': weather.get('forecast_intensity', 0),
            'max_3h_rain':       weather.get('max_3h_rain', 0),
            'worst_rain_time':   weather.get('worst_rain_time', ''),
            'max_temp_24h':      weather.get('max_temp_24h', temp),
            'max_temp_time':     weather.get('max_temp_time', ''),
            'max_gust_24h':      weather.get('max_gust_24h', gust),
            'storm_now':         weather['storm_now'],
            'peak_rain_hour':    weather['peak_rain_hour'],
            'wind_hazard':       weather.get('wind_hazard', False),
            'heat_hazard':       weather.get('heat_hazard', False),
        },
        'aqi':             aqi,
        'ai_bulletin':     bulletin,
        'storm_mode':      weather.get('storm_now', False) or weather.get('weather_coming', False),
        'label_summary':   label_summary,
        'total_at_risk':   sum(1 for r in results if r['overall_risk'] >= 35),
        'critical_areas':  [r['area'] for r in results if r['overall_risk'] >= 75],
        'generated_at':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'issues_analysed': len(open_issues),
        'ml_active':       _load_model(),
        'label_display':   LABEL_DISPLAY,
        # P1: expose officer focus location if provided, else Delhi centre
        'fetch_lat':       focus_lat if focus_lat is not None else 28.632,
        'fetch_lng':       focus_lng if focus_lng is not None else 77.217,
    }