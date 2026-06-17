"""
AreaPulse CivicAlert Engine v4.4
==================================
Changes from v4.3:
  - [P4] WeatherAPI.com as primary weather source (accurate, IMD-backed alerts)
  - [P4] Open-Meteo kept as automatic fallback if WeatherAPI fails/unavailable
  - [P4] parse_live_weather_weatherapi() — new parser for WeatherAPI JSON
  - [P4] parse_live_weather_openmeteo() — renamed from parse_live_weather()
  - [P4] fetch_live_weather() — tries WeatherAPI first, falls back silently
  - [P4] parse_live_weather() — unified entry point, routes to correct parser
  - [P4] Both parsers produce identical 39-key output dict (no downstream changes)
  - [P4] WeatherAPI: real IMD thunderstorm alerts, accurate temp, real condition text
  - All v4.3 fixes preserved (WMO dict, hourly_precip_12h, past_hours lag fix)
  - score_area, generate_bulletin, run_full_prediction — UNCHANGED

Requires env var: WEATHER_API_KEY (from weatherapi.com free tier)
Fallback: Open-Meteo (free, no key needed) — auto-activates if WEATHER_API_KEY missing
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
# drain:    0=none, 1=poor, 2=moderate, 3=good
# elev:     0=low/flood-prone, 1=mid, 2=high/safe
# road_age: 0=new(<2yr), 1=5yr, 2=10yr, 3=old(>15yr)
# infra_age:0=new, 1=5yr, 2=10yr, 3=old
# wp:       0=low, 1=medium, 2=high waterlogging potential
# pop:      0=sparse, 1=medium, 2=dense
# zone:     NORTH/SOUTH/EAST/WEST/CENTRE/NW/NE/SW/SE

DELHI_AREAS = {
    # ── CENTRE ────────────────────────────────────────────────────
    'Chandni Chowk':      {'lat':28.6506,'lng':77.2334,'drain':0,'elev':0,'road_age':3,'infra_age':3,'wp':2,'pop':2,'zone':'CENTRE'},
    'Kashmere Gate':      {'lat':28.6671,'lng':77.2276,'drain':0,'elev':0,'road_age':3,'infra_age':3,'wp':2,'pop':2,'zone':'CENTRE'},
    'Paharganj':          {'lat':28.6448,'lng':77.2167,'drain':0,'elev':0,'road_age':3,'infra_age':3,'wp':2,'pop':2,'zone':'CENTRE'},
    'Connaught Place':    {'lat':28.6315,'lng':77.2167,'drain':2,'elev':1,'road_age':2,'infra_age':2,'wp':2,'pop':1,'zone':'CENTRE'},
    'Karol Bagh':         {'lat':28.6514,'lng':77.1907,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'CENTRE'},
    'Patel Nagar':        {'lat':28.6548,'lng':77.1630,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'CENTRE'},
    'Daryaganj':          {'lat':28.6420,'lng':77.2440,'drain':0,'elev':0,'road_age':3,'infra_age':3,'wp':2,'pop':2,'zone':'CENTRE'},
    'Jama Masjid':        {'lat':28.6507,'lng':77.2334,'drain':0,'elev':0,'road_age':3,'infra_age':3,'wp':2,'pop':2,'zone':'CENTRE'},
    'New Delhi Station':  {'lat':28.6421,'lng':77.2197,'drain':1,'elev':0,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'CENTRE'},
    'Minto Road':         {'lat':28.6384,'lng':77.2290,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':1,'pop':1,'zone':'CENTRE'},
    'Lodhi Colony':       {'lat':28.5931,'lng':77.2257,'drain':3,'elev':2,'road_age':1,'infra_age':1,'wp':1,'pop':0,'zone':'CENTRE'},
    'Sarojini Nagar':     {'lat':28.5765,'lng':77.1954,'drain':2,'elev':1,'road_age':2,'infra_age':2,'wp':1,'pop':1,'zone':'CENTRE'},
    'INA':                {'lat':28.5741,'lng':77.2092,'drain':2,'elev':1,'road_age':1,'infra_age':1,'wp':1,'pop':1,'zone':'CENTRE'},

    # ── NORTH ─────────────────────────────────────────────────────
    'Civil Lines':        {'lat':28.6800,'lng':77.2230,'drain':2,'elev':1,'road_age':2,'infra_age':2,'wp':1,'pop':1,'zone':'NORTH'},
    'Mukherjee Nagar':    {'lat':28.7040,'lng':77.2080,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':1,'pop':2,'zone':'NORTH'},
    'Model Town':         {'lat':28.7120,'lng':77.1900,'drain':2,'elev':1,'road_age':1,'infra_age':2,'wp':1,'pop':1,'zone':'NORTH'},
    'Pitampura':          {'lat':28.7010,'lng':77.1320,'drain':2,'elev':1,'road_age':1,'infra_age':1,'wp':1,'pop':2,'zone':'NORTH'},
    'Burari':             {'lat':28.7380,'lng':77.2010,'drain':1,'elev':0,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'NORTH'},
    'Timarpur':           {'lat':28.7150,'lng':77.2110,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':1,'pop':2,'zone':'NORTH'},
    'Bara Hindu Rao':     {'lat':28.6720,'lng':77.2150,'drain':0,'elev':0,'road_age':3,'infra_age':3,'wp':2,'pop':2,'zone':'NORTH'},

    # ── NORTH-WEST ────────────────────────────────────────────────
    'Rohini':             {'lat':28.7480,'lng':77.0670,'drain':2,'elev':1,'road_age':1,'infra_age':1,'wp':1,'pop':2,'zone':'NW'},
    'Shalimar Bagh':      {'lat':28.7200,'lng':77.1500,'drain':2,'elev':1,'road_age':1,'infra_age':1,'wp':1,'pop':2,'zone':'NW'},
    'Ashok Vihar':        {'lat':28.6950,'lng':77.1750,'drain':2,'elev':1,'road_age':1,'infra_age':2,'wp':1,'pop':2,'zone':'NW'},
    'Wazirpur':           {'lat':28.6900,'lng':77.1600,'drain':1,'elev':0,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'NW'},
    'Netaji Subhash Pl':  {'lat':28.6950,'lng':77.1400,'drain':2,'elev':1,'road_age':1,'infra_age':1,'wp':1,'pop':2,'zone':'NW'},
    'Narela':             {'lat':28.8510,'lng':77.0940,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':1,'pop':1,'zone':'NW'},
    'Bawana':             {'lat':28.7990,'lng':77.0380,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':1,'pop':1,'zone':'NW'},
    'Alipur':             {'lat':28.7970,'lng':77.1340,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':1,'pop':1,'zone':'NW'},

    # ── WEST ──────────────────────────────────────────────────────
    'Punjabi Bagh':       {'lat':28.6708,'lng':77.1311,'drain':2,'elev':1,'road_age':2,'infra_age':2,'wp':1,'pop':1,'zone':'WEST'},
    'Rajouri Garden':     {'lat':28.6465,'lng':77.1150,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'WEST'},
    'Janakpuri':          {'lat':28.6219,'lng':77.0878,'drain':2,'elev':1,'road_age':2,'infra_age':1,'wp':1,'pop':2,'zone':'WEST'},
    'Dwarka':             {'lat':28.5823,'lng':77.0500,'drain':2,'elev':1,'road_age':1,'infra_age':1,'wp':1,'pop':2,'zone':'WEST'},
    'Uttam Nagar':        {'lat':28.6210,'lng':77.0590,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'WEST'},
    'Vikaspuri':          {'lat':28.6350,'lng':77.0710,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'WEST'},
    'Tilak Nagar':        {'lat':28.6440,'lng':77.1010,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'WEST'},
    'Subhash Nagar':      {'lat':28.6380,'lng':77.1120,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':1,'pop':2,'zone':'WEST'},
    'Paschim Vihar':      {'lat':28.6710,'lng':77.0900,'drain':2,'elev':1,'road_age':1,'infra_age':1,'wp':1,'pop':2,'zone':'WEST'},
    'Nangloi':            {'lat':28.6720,'lng':77.0580,'drain':0,'elev':0,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'WEST'},
    'Nihal Vihar':        {'lat':28.6760,'lng':77.0700,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':1,'pop':2,'zone':'WEST'},

    # ── NORTH-EAST ────────────────────────────────────────────────
    'Shahdara':           {'lat':28.6695,'lng':77.2993,'drain':0,'elev':0,'road_age':3,'infra_age':3,'wp':2,'pop':2,'zone':'NE'},
    'Preet Vihar':        {'lat':28.6455,'lng':77.2927,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':1,'pop':2,'zone':'NE'},
    'Vivek Vihar':        {'lat':28.6720,'lng':77.3150,'drain':1,'elev':1,'road_age':1,'infra_age':1,'wp':1,'pop':2,'zone':'NE'},
    'Gandhi Nagar':       {'lat':28.6620,'lng':77.2760,'drain':0,'elev':0,'road_age':3,'infra_age':3,'wp':2,'pop':2,'zone':'NE'},
    'Geeta Colony':       {'lat':28.6530,'lng':77.2780,'drain':0,'elev':0,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'NE'},
    'Anand Vihar':        {'lat':28.6470,'lng':77.3160,'drain':1,'elev':1,'road_age':1,'infra_age':1,'wp':1,'pop':2,'zone':'NE'},

    # ── EAST ──────────────────────────────────────────────────────
    'Laxmi Nagar':        {'lat':28.6330,'lng':77.2780,'drain':1,'elev':0,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'EAST'},
    'Mayur Vihar':        {'lat':28.6096,'lng':77.2946,'drain':2,'elev':1,'road_age':1,'infra_age':1,'wp':1,'pop':2,'zone':'EAST'},
    'Patparganj':         {'lat':28.6230,'lng':77.2990,'drain':1,'elev':1,'road_age':1,'infra_age':1,'wp':1,'pop':2,'zone':'EAST'},
    'IP Extension':       {'lat':28.6280,'lng':77.3120,'drain':1,'elev':1,'road_age':1,'infra_age':1,'wp':1,'pop':2,'zone':'EAST'},
    'Kondli':             {'lat':28.5980,'lng':77.3160,'drain':0,'elev':0,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'EAST'},

    # ── SOUTH-EAST ────────────────────────────────────────────────
    'Nizamuddin':         {'lat':28.5890,'lng':77.2480,'drain':1,'elev':0,'road_age':2,'infra_age':3,'wp':2,'pop':2,'zone':'SE'},
    'Lajpat Nagar':       {'lat':28.5700,'lng':77.2433,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'SE'},
    'Defence Colony':     {'lat':28.5740,'lng':77.2310,'drain':3,'elev':2,'road_age':1,'infra_age':1,'wp':1,'pop':0,'zone':'SE'},
    'Greater Kailash':    {'lat':28.5494,'lng':77.2436,'drain':3,'elev':2,'road_age':1,'infra_age':1,'wp':1,'pop':1,'zone':'SE'},
    'Nehru Place':        {'lat':28.5491,'lng':77.2543,'drain':2,'elev':1,'road_age':1,'infra_age':1,'wp':1,'pop':1,'zone':'SE'},
    'Kalkaji':            {'lat':28.5366,'lng':77.2590,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':1,'pop':2,'zone':'SE'},
    'Okhla':              {'lat':28.5244,'lng':77.2860,'drain':0,'elev':0,'road_age':2,'infra_age':2,'wp':1,'pop':2,'zone':'SE'},
    'Sarita Vihar':       {'lat':28.5280,'lng':77.2930,'drain':1,'elev':1,'road_age':1,'infra_age':1,'wp':1,'pop':2,'zone':'SE'},
    'Badarpur':           {'lat':28.5000,'lng':77.2930,'drain':0,'elev':0,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'SE'},
    'Tughlakabad':        {'lat':28.5050,'lng':77.2600,'drain':0,'elev':0,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'SE'},

    # ── SOUTH ─────────────────────────────────────────────────────
    'Saket':              {'lat':28.5245,'lng':77.2066,'drain':3,'elev':2,'road_age':1,'infra_age':1,'wp':1,'pop':1,'zone':'SOUTH'},
    'Malviya Nagar':      {'lat':28.5355,'lng':77.2010,'drain':2,'elev':1,'road_age':1,'infra_age':1,'wp':1,'pop':1,'zone':'SOUTH'},
    'Hauz Khas':          {'lat':28.5494,'lng':77.2001,'drain':3,'elev':2,'road_age':1,'infra_age':1,'wp':1,'pop':1,'zone':'SOUTH'},
    'Mehrauli':           {'lat':28.5244,'lng':77.1855,'drain':0,'elev':0,'road_age':3,'infra_age':3,'wp':1,'pop':2,'zone':'SOUTH'},
    'Vasant Kunj':        {'lat':28.5205,'lng':77.1575,'drain':3,'elev':2,'road_age':1,'infra_age':1,'wp':1,'pop':1,'zone':'SOUTH'},
    'Chhatarpur':         {'lat':28.4980,'lng':77.1740,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':1,'pop':1,'zone':'SOUTH'},
    'Sangam Vihar':       {'lat':28.5130,'lng':77.2200,'drain':0,'elev':0,'road_age':2,'infra_age':2,'wp':2,'pop':2,'zone':'SOUTH'},
    'Ambedkar Nagar':     {'lat':28.5220,'lng':77.2130,'drain':1,'elev':1,'road_age':2,'infra_age':2,'wp':1,'pop':2,'zone':'SOUTH'},

    # ── SOUTH-WEST ────────────────────────────────────────────────
    'Vasant Vihar':       {'lat':28.5621,'lng':77.1567,'drain':3,'elev':2,'road_age':1,'infra_age':1,'wp':1,'pop':0,'zone':'SW'},
    'RK Puram':           {'lat':28.5649,'lng':77.1700,'drain':3,'elev':2,'road_age':1,'infra_age':1,'wp':1,'pop':1,'zone':'SW'},
}

# ── RISK LABELS ───────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════
#  WEATHER — PRIMARY: WeatherAPI.com  FALLBACK: Open-Meteo
# ══════════════════════════════════════════════════════════════

# WeatherAPI condition codes for storm flag detection
_WAPI_THUNDER_CODES = {1087, 1273, 1276, 1279, 1282}
_WAPI_HEAVY_RAIN    = {1192, 1195, 1201, 1243, 1246}
_WAPI_MOD_RAIN      = {1063, 1180, 1183, 1186, 1189}
_WAPI_LIGHT_RAIN    = {1150, 1153, 1168, 1171, 1198}
_WAPI_DRIZZLE       = {1150, 1153}
_WAPI_FOG           = {1030, 1135, 1147}
_WAPI_ALL_RAIN      = _WAPI_THUNDER_CODES | _WAPI_HEAVY_RAIN | _WAPI_MOD_RAIN | _WAPI_LIGHT_RAIN

# Open-Meteo WMO codes — kept for fallback parser
_WMO = {
    0:'Clear sky', 1:'Mainly clear', 2:'Partly cloudy', 3:'Overcast',
    45:'Fog', 48:'Rime fog',
    51:'Light drizzle', 53:'Moderate drizzle', 55:'Dense drizzle',
    56:'Light freezing drizzle', 57:'Heavy freezing drizzle',
    61:'Slight rain', 63:'Moderate rain', 65:'Heavy rain',
    66:'Light freezing rain', 67:'Heavy freezing rain',
    71:'Slight snow', 73:'Moderate snow', 75:'Heavy snow', 77:'Snow grains',
    80:'Slight showers', 81:'Moderate showers', 82:'Violent showers',
    85:'Slight snow showers', 86:'Heavy snow showers',
    95:'Thunderstorm', 96:'Thunderstorm with hail', 99:'Severe thunderstorm',
}


def _fetch_weatherapi(lat, lng, api_key):
    """Fetch from WeatherAPI.com — returns raw JSON or None."""
    try:
        import requests as _req
        r = _req.get(
            'https://api.weatherapi.com/v1/forecast.json',
            params={
                'key':   api_key,
                'q':     f'{lat},{lng}',
                'days':  2,
                'aqi':   'no',
                'alerts':'yes',
            },
            timeout=(3, 8),
        )
        r.raise_for_status()
        data = r.json()
        print(f"[weather] WeatherAPI OK — {data['current']['condition']['text']}, {data['current']['temp_c']}°C")
        return {'source': 'weatherapi', 'data': data}
    except Exception as e:
        print(f"[weather] WeatherAPI failed: {e}")
        return None


def _fetch_openmeteo(lat, lng):
    """Fetch from Open-Meteo — returns raw JSON or None."""
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
            timeout=(3, 8),
        )
        r.raise_for_status()
        print(f"[weather] Open-Meteo OK (fallback)")
        return {'source': 'openmeteo', 'data': r.json()}
    except Exception as e:
        print(f"[weather] Open-Meteo failed: {e}")
        return None


def fetch_live_weather(lat, lng):
    """
    Try WeatherAPI first (accurate, IMD alerts).
    Falls back to Open-Meteo automatically if key missing or request fails.
    Returns dict with 'source' and 'data' keys, or None if both fail.
    """
    api_key = os.environ.get('WEATHER_API_KEY', '').strip()
    if api_key:
        result = _fetch_weatherapi(lat, lng, api_key)
        if result is not None:
            return result
        print("[weather] WeatherAPI failed — falling back to Open-Meteo")
    else:
        print("[weather] WEATHER_API_KEY not set — using Open-Meteo")

    return _fetch_openmeteo(lat, lng)


def _parse_weatherapi(raw):
    """
    Parse WeatherAPI.com JSON into standard 39-key weather dict.
    Identical output shape to _parse_openmeteo() so all downstream code works unchanged.
    """
    c    = raw.get('current', {})
    forecastdays = raw.get('forecast', {}).get('forecastday', [{}])
    fc   = forecastdays[0]                      # today
    fc2  = forecastdays[1] if len(forecastdays) > 1 else {}  # tomorrow
    hrs  = fc.get('hour', [])
    hrs2 = fc2.get('hour', [])   # tomorrow's hours for max temp/gust
    alts = raw.get('alerts', {}).get('alert', [])

    curr_temp  = float(c.get('temp_c')       or 25)
    curr_rain  = float(c.get('precip_mm')    or 0)
    curr_wind  = float(c.get('wind_kph')     or 0)
    curr_gust  = float(c.get('gust_kph')     or 0)
    curr_humid = float(c.get('humidity')     or 60)
    curr_press = float(c.get('pressure_mb')  or 1010)
    curr_vis   = float(c.get('vis_km', 9.9)  or 9.9) * 1000   # km → m
    cond_text  = c.get('condition', {}).get('text', 'Unknown')
    cond_code  = int(c.get('condition', {}).get('code') or 1000)

    # Hourly data — WeatherAPI gives full 24h for today
    def _hf(key, d=0.0):
        return [float(h.get(key) or d) for h in hrs]

    hp  = _hf('precip_mm')         # hourly precip
    ht  = _hf('temp_c', 25)        # hourly temp
    hg  = _hf('gust_kph')          # hourly gust
    hpr = _hf('pressure_mb', 1010) # hourly pressure
    hcc = [int(h.get('condition', {}).get('code') or 1000) for h in hrs]

    # Current-only precip is often 0 even mid-rain — use last hour max
    if hrs:
        now_hour = datetime.now().hour
        recent_precip = max(
            float(hrs[now_hour].get('precip_mm') or 0) if now_hour < len(hrs) else 0,
            float(hrs[max(0, now_hour-1)].get('precip_mm') or 0) if now_hour > 0 else 0,
        )
        curr_rain = max(curr_rain, recent_precip)

    # Storm flags from condition code
    thunder_now = cond_code in _WAPI_THUNDER_CODES
    rain_now    = cond_code in _WAPI_ALL_RAIN
    fog_now     = cond_code in _WAPI_FOG
    wind_hazard = curr_gust >= 40
    # heat_hazard: current temp OR today's forecast max (catches morning/evening readings)
    day_max_c   = float(fc.get('day', {}).get('maxtemp_c') or 0)
    heat_hazard = curr_temp >= 40 or day_max_c >= 40
    storm_now   = thunder_now or rain_now or wind_hazard or (curr_temp >= 40) or fog_now
    # Note: heat_hazard is True even if storm_now=False — used for forecast bar

    # IMD alerts = FORECAST WARNING only, not live condition.
    # Only condition code tells us what is happening RIGHT NOW.
    # Alerts set thunder_soon (used for forecast bar), never thunder_now.
    has_alert_thunder = any(
        'thunder' in str(a.get('headline', '')).lower() or
        'thunder' in str(a.get('event', '')).lower()
        for a in alts
    )
    if has_alert_thunder:
        print(f"[weather] IMD alert (forecast, not live): {(alts[0].get('headline','') or alts[0].get('event',''))[:80]}")

    # WMO-equivalent code for ML model (maps WeatherAPI → approximate WMO)
    if cond_code in _WAPI_THUNDER_CODES:   curr_code = 95
    elif cond_code in _WAPI_HEAVY_RAIN:    curr_code = 65
    elif cond_code in _WAPI_MOD_RAIN:      curr_code = 63
    elif cond_code in _WAPI_LIGHT_RAIN:    curr_code = 61
    elif cond_code in _WAPI_DRIZZLE:       curr_code = 51
    elif cond_code in _WAPI_FOG:           curr_code = 45
    elif curr_temp >= 40:                  curr_code = 3
    else:                                  curr_code = 1

    weather_intensity = min(1.0, (
        (curr_code / 99) * 0.5 +
        (max(0, curr_rain) / 20) * 0.3 +
        (max(0, curr_gust - 20) / 80) * 0.1 +
        (max(0, curr_temp - 30) / 20) * 0.1
    ))

    # Rain horizons
    rain_1h = curr_rain
    rain_3h = sum(hp[:3])  if len(hp) >= 3  else curr_rain * 3
    rain_6h = sum(hp[:6])  if len(hp) >= 6  else curr_rain * 6
    rain_24h= sum(hp[:24]) if len(hp) >= 24 else curr_rain * 24

    # Pressure trend
    press_trend = (hpr[0] - hpr[2]) if len(hpr) >= 3 else 0

    # Peak rain window
    max_3h_rain    = max((sum(hp[i:i+3]) for i in range(0, min(21, len(hp)-2))), default=0)
    worst_3h_start = max(range(0, min(21, len(hp)-2)), key=lambda i: sum(hp[i:i+3]), default=0) if len(hp) >= 3 else 0
    worst_rain_time = (datetime.now() + timedelta(hours=worst_3h_start)).strftime('%I:%M %p')
    _wrd = datetime.now() + timedelta(hours=worst_3h_start)
    worst_rain_day  = 'Today' if worst_3h_start <= 2 else _wrd.strftime('%a %d %b')

    # Max temp and gust — use today remaining + tomorrow to cover full 24h window
    # WeatherAPI hourly is midnight-to-midnight; at 2pm only 10 hours remain today
    now_h = datetime.now().hour
    ht2   = [float(h.get('temp_c') or 25) for h in hrs2]  if hrs2 else []
    hg2   = [float(h.get('gust_kph') or 0) for h in hrs2] if hrs2 else []
    # remaining today from now_h onwards + first (24-remaining) hours of tomorrow
    remaining_today = ht[now_h:]
    fill_tomorrow   = ht2[:max(0, 24 - len(remaining_today))]
    temp_window     = remaining_today + fill_tomorrow
    gust_window_raw = hg[now_h:] + hg2[:max(0, 24 - len(hg[now_h:]))]

    # Also use fc day max as authoritative daily max (WeatherAPI day summary)
    day_max_temp = float(fc.get('day', {}).get('maxtemp_c') or 0)
    max_temp_24h  = max(max(temp_window) if temp_window else curr_temp, day_max_temp)
    max_temp_hour_idx = temp_window.index(max(temp_window)) if temp_window else 0
    max_temp_time = (datetime.now() + timedelta(hours=max_temp_hour_idx)).strftime('%I:%M %p')
    max_gust_24h  = max(max(gust_window_raw) if gust_window_raw else curr_gust,
                        float(fc.get('day', {}).get('maxwind_kph') or 0))

    # Thunderstorm forecast from hourly codes
    thunder_soon = has_alert_thunder or any(c in _WAPI_THUNDER_CODES for c in hcc[:24])
    thunder_hour = next((i for i, c in enumerate(hcc[:24]) if c in _WAPI_THUNDER_CODES), None)
    if thunder_soon and not has_alert_thunder and thunder_hour is not None:
        thunder_time = (datetime.now() + timedelta(hours=thunder_hour)).strftime('%I:%M %p')
    elif has_alert_thunder:
        thunder_time = 'active now'
    else:
        thunder_time = None

    # Forecast intensity
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

    # Forecast summary
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

    peak_i  = hp[:24].index(max(hp[:24])) if hp else 0
    peak_hr = 'NOW' if storm_now else (datetime.now() + timedelta(hours=peak_i)).strftime('%I:%M %p')

    return {
        'curr_rain':  round(curr_rain,  1),
        'curr_code':  curr_code,
        'curr_temp':  round(curr_temp,  1),
        'curr_wind':  round(curr_wind,  1),
        'curr_gust':  round(curr_gust,  1),
        'curr_humid': round(curr_humid, 1),
        'curr_press': round(curr_press, 1),
        'curr_vis':   round(curr_vis,   0),
        'press_trend': round(press_trend, 2),
        'current_condition': cond_text,   # direct text, e.g. "Partly cloudy"

        'rain_next_1h': round(rain_1h,  1),
        'rain_next_3h': round(rain_3h,  1),
        'rain_next_6h': round(rain_6h,  1),
        'rain_24h':     round(rain_24h, 1),

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

        'max_3h_rain':       round(max_3h_rain,   1),
        'worst_rain_time':   worst_rain_time,
        'worst_rain_day':    worst_rain_day,
        'max_temp_24h':      round(max_temp_24h,  1),
        'max_temp_time':     max_temp_time,
        'max_gust_24h':      round(max_gust_24h,  1),
        'weather_coming':    weather_coming,
        'forecast_intensity': forecast_intensity,
        'forecast_summary':  forecast_summary,

        'hourly_precip_12h': [round(x, 1) for x in hp[:12]],
        'hourly_codes_12h':  hcc[:12],

        'month': datetime.now().month,
        'hour':  datetime.now().hour,
        '_source': 'weatherapi',
    }


def _parse_openmeteo(data):
    """
    Parse Open-Meteo JSON into standard 39-key weather dict.
    Identical to v4.3 parse_live_weather() — preserved exactly.
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

    # past_hours=2 — fix: only use current for display, not past max
    # Use current.precipitation directly; past hours only for scoring
    past_rain_for_score = max(hp[:2]) if len(hp) >= 2 else 0
    past_code_for_score = max(hc[:2]) if len(hc) >= 2 else 0
    # Only update curr_rain if current API says 0 but last hour had significant rain
    if curr_rain == 0 and past_rain_for_score > 0.5:
        curr_rain = past_rain_for_score * 0.5  # decay — it was raining, may still be
    # Only bump code if current is clear but past had active weather
    if curr_code == 0 and past_code_for_score >= 61:
        curr_code = past_code_for_score

    forecast_hp = hp[2:] if len(hp) > 2 else hp
    forecast_hc = hc[2:] if len(hc) > 2 else hc

    rain_24h    = sum(forecast_hp[:24]) if forecast_hp else 0
    press_trend = (hpr[0] - hpr[2]) if len(hpr) >= 3 else 0
    rain_1h     = curr_rain
    rain_3h     = sum(forecast_hp[:3]) if len(forecast_hp) >= 3 else curr_rain * 3
    rain_6h     = sum(forecast_hp[:6]) if len(forecast_hp) >= 6 else curr_rain * 6

    hp = forecast_hp
    hc = forecast_hc

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
    worst_rain_day  = 'Today' if worst_3h_start <= 2 else _wrd.strftime('%a %d %b')

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
        'curr_rain':  round(curr_rain,  1),
        'curr_code':  curr_code,
        'curr_temp':  round(curr_temp,  1),
        'curr_wind':  round(curr_wind,  1),
        'curr_gust':  round(curr_gust,  1),
        'curr_humid': round(curr_humid, 1),
        'curr_press': round(curr_press, 1),
        'curr_vis':   round(curr_vis,   0),
        'press_trend': round(press_trend, 2),
        'current_condition': _WMO.get(curr_code, 'Unknown'),

        'rain_next_1h': round(rain_1h,  1),
        'rain_next_3h': round(rain_3h,  1),
        'rain_next_6h': round(rain_6h,  1),
        'rain_24h':     round(rain_24h, 1),

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

        'max_3h_rain':       round(max_3h_rain,   1),
        'worst_rain_time':   worst_rain_time,
        'worst_rain_day':    worst_rain_day,
        'max_temp_24h':      round(max_temp_24h,  1),
        'max_temp_time':     max_temp_time,
        'max_gust_24h':      round(max_gust_24h,  1),
        'weather_coming':    weather_coming,
        'forecast_intensity': forecast_intensity,
        'forecast_summary':  forecast_summary,

        'hourly_precip_12h': [round(x, 1) for x in hp[:12]],
        'hourly_codes_12h':  hc[:12],

        'month': datetime.now().month,
        'hour':  datetime.now().hour,
        '_source': 'openmeteo',
    }


def parse_live_weather(raw_result):
    """
    Unified entry point. Routes to WeatherAPI or Open-Meteo parser
    based on source tag. Identical output shape from both parsers.
    """
    if raw_result is None:
        return None
    source = raw_result.get('source', 'openmeteo')
    data   = raw_result.get('data')
    if source == 'weatherapi':
        return _parse_weatherapi(data)
    else:
        return _parse_openmeteo(data)


# ── AQI ───────────────────────────────────────────────────────
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

            vuln = (
                (3 - meta['drain'])    * 0.30 +
                (3 - meta['elev'])     * 0.15 +
                meta['road_age']       * 0.20 +
                meta['infra_age']      * 0.20 +
                meta['pop']            * 0.15
            ) / 3.0

            for li, label in enumerate(LABELS):
                if li < len(probs):
                    prob = float(probs[li][0][1]) if probs[li].shape[1] > 1 else float(probs[li][0][0])
                    scores[label] = min(100, round(prob * 100 * (0.6 + 0.4 * vuln)))
                else:
                    scores[label] = 0

        except Exception as e:
            print(f"[score] ML failed for {area_name}: {e}")

    if not scores:
        rain = eff_rain
        scores = {
            'label_flood':           min(100, round((rain / 20) * 80 * (1 + (2 - meta['drain']) * 0.3))),
            'label_pothole_worsen':  min(100, round((rain / 15) * 60 + p_open * 8)),
            'label_sewage_overflow': min(100, round((rain / 25) * 90 * (1 + (2 - meta['drain']) * 0.4) + s_open * 10)),
            'label_garbage_flood':   min(100, round((rain / 20) * 50 + g_open * 6)),
            'label_elec_hazard':     min(100, round((eff_gust / 60) * 70 + (50 if eff_thunder else 0) + e_open * 8)),
        }

    overall_risk = round(sum(scores.values()) / max(len(scores), 1))

    top_risk    = max(scores, key=scores.get) if scores else 'label_flood'
    top_score   = scores.get(top_risk, 0)
    tti_base    = max(0, 60 - overall_risk)
    time_to_impact_mins = tti_base if not using_forecast else max(30, tti_base * 2)

    drain_labels = {0: 'None', 1: 'Poor', 2: 'Moderate', 3: 'Good'}
    return {
        'area':               area_name,
        'lat':                meta['lat'],
        'lng':                meta['lng'],
        'scores':             scores,
        'overall_risk':       overall_risk,
        'top_risk':           top_risk,
        'top_score':          top_score,
        'time_to_impact_mins': time_to_impact_mins,
        'open_issues':        len(area_issues),
        'open_issues_count':  len(area_issues),
        'issue_ids':          [str(i.get('id', '')) for i in area_issues[:5]],
        'drain_quality':      drain_labels.get(meta['drain'], 'Unknown'),
        'zone':               meta.get('zone', 'CENTRE'),
        'affected_issues':    [
            {
                'id':         str(i.get('id', '')),
                'tag':        i.get('tag', ''),
                'severity':   i.get('severity', 'low'),
                'description': str(i.get('description', '')),
                'risk_label': top_risk,
            }
            for i in area_issues[:10]
        ],
    }


# ── BULLETIN ──────────────────────────────────────────────────
def generate_bulletin(results, weather, aqi, groq_api_key):
    top5 = results[:5]

    rain  = weather['curr_rain']
    temp  = weather['curr_temp']
    cond  = weather['current_condition']
    gust  = weather.get('curr_gust', 0)
    fcst  = weather.get('forecast_summary', '')
    storm = weather.get('storm_now', False)

    fallback_parts = []
    if storm:
        fallback_parts.append(f"Active weather event: {cond}.")
    if rain > 0:
        fallback_parts.append(f"{rain}mm/hr rainfall detected.")
    if top5:
        t  = top5[0]
        tr = t.get('top_risk', '')
        ids = t.get('issue_ids', [])
        fallback_parts.append(
            f"Highest risk: {t['area']} ({t['overall_risk']}/100) — "
            f"{LABEL_DISPLAY.get(tr, {}).get('name', 'flooding')} in ~{t['time_to_impact_mins']}min. "
            f"{'Issues ' + ', '.join(ids) + ' will worsen. ' if ids else ''}"
            f"Deploy {LABEL_DISPLAY.get(tr, {}).get('dept', 'relevant dept')} immediately."
        )
    if not fallback_parts:
        fallback_parts.append(f"{cond} — {temp}°C. {fcst}. No immediate civic risk.")

    fallback = ' '.join(fallback_parts)

    if not groq_api_key:
        return fallback

    try:
        import requests as _req
        area_lines = '\n'.join(
            f"- {r['area']}: risk={r['overall_risk']}/100, top={r.get('top_risk','')}, issues={r['open_issues']}"
            for r in top5
        )
        aqi_str = f"AQI {aqi['aqi']} ({aqi.get('source','')})" if aqi else "AQI unavailable"
        prompt = (
            f"You are AreaPulse CivicAlert AI. Generate a 2-sentence operational bulletin for Delhi municipal officers.\n\n"
            f"Weather: {cond}, {temp}°C, rain={rain}mm/hr, gusts={gust}km/h. {fcst}\n"
            f"{aqi_str}\n"
            f"Top risk areas:\n{area_lines}\n\n"
            f"Write 2 sentences: (1) current situation, (2) recommended action. Be specific. No markdown."
        )
        resp = _req.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization': f'Bearer {groq_api_key}', 'Content-Type': 'application/json'},
            json={
                'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 120,
                'temperature': 0.3,
            },
            timeout=(3, 10),
        )
        resp.raise_for_status()
        text = resp.json()['choices'][0]['message']['content'].strip()
        print(f"[bulletin] Groq OK ({len(text)} chars)")
        return text
    except Exception as e:
        print(f"[bulletin] Groq failed: {e} — using fallback")
        return fallback


# ── MAIN ENTRY POINT ──────────────────────────────────────────
def run_full_prediction(open_issues=None, groq_api_key=None, aqi_token='demo',
                        focus_lat=None, focus_lng=None, rain_override_mm=0):
    """
    Run full civic risk prediction for all Delhi areas.
    focus_lat / focus_lng — fetch weather for user's exact location.
    rain_override_mm      — manual rain override (0=off, 5=light, 25=heavy, 40=storm).
    """
    open_issues = open_issues or []
    _load_model()
    api_key = os.environ.get('WEATHER_API_KEY', '').strip()
    src_label = 'WeatherAPI' if api_key else 'Open-Meteo'
    print(f"[engine] v4.5 · {len(DELHI_AREAS)} areas · {len(open_issues)} issues · source={src_label}"
          + (f" · rain_override={rain_override_mm}mm" if rain_override_mm else ""))

    ZONES = {
        'centre': (28.632, 77.217),   # Connaught Place
        'north':  (28.720, 77.200),   # Civil Lines / Burari
        'nw':     (28.730, 77.090),   # Rohini / Shalimar Bagh
        'west':   (28.630, 77.080),   # Janakpuri / Uttam Nagar
        'sw':     (28.560, 77.160),   # Vasant Vihar / RK Puram
        'south':  (28.510, 77.210),   # Saket / Sangam Vihar
        'se':     (28.540, 77.270),   # Kalkaji / Badarpur
        'east':   (28.630, 77.310),   # Mayur Vihar / Patparganj
        'ne':     (28.660, 77.300),   # Shahdara / Vivek Vihar
    }
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
        raise Exception("All weather sources unreachable for all zones.")

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
        alat, alng = meta['lat'], meta['lng']
        best_zone, best_d = 'centre', float('inf')
        for zn, (zlat, zlng) in zone_coords.items():
            d = ((alat - zlat)**2 + (alng - zlng)**2)**0.5
            if d < best_d:
                best_d, best_zone = d, zn
        return zone_weather.get(best_zone) or zone_weather.get('centre')

    if focus_lat is not None and focus_lng is not None and zone_weather.get('officer'):
        weather = zone_weather['officer']
        print(f"[engine] Using officer location weather: rain={weather.get('curr_rain')} cond={weather.get('current_condition')}")
    else:
        weather = zone_weather.get('centre') or next(
            w for w in zone_weather.values() if w is not None
        )

    # Rain override
    WMO_OVERRIDE = {5: 61, 25: 65, 40: 95}
    if rain_override_mm > 0:
        override_code = WMO_OVERRIDE.get(rain_override_mm, 63)
        print(f"[engine] Rain override: {rain_override_mm}mm → code={override_code}")
        for zn in zone_weather:
            if zone_weather[zn] is None:
                continue
            zone_weather[zn] = dict(zone_weather[zn])
            zone_weather[zn]['curr_rain']    = float(rain_override_mm)
            zone_weather[zn]['curr_code']    = override_code
            zone_weather[zn]['rain_next_1h'] = float(rain_override_mm)
            zone_weather[zn]['rain_next_3h'] = float(rain_override_mm) * 2
            zone_weather[zn]['rain_next_6h'] = float(rain_override_mm) * 3
            zone_weather[zn]['rain_now']     = True
            zone_weather[zn]['storm_now']    = rain_override_mm >= 15
            zone_weather[zn]['thunder_now']  = rain_override_mm >= 40
        if focus_lat is not None and zone_weather.get('officer'):
            weather = zone_weather['officer']
        else:
            weather = zone_weather.get('centre') or next(
                w for w in zone_weather.values() if w is not None
            )

    print("[engine] Fetching AQI...")
    aqi = fetch_aqi(aqi_token)

    results = []
    for area_name, meta in DELHI_AREAS.items():
        area_wx = _zone_for_area(meta)
        results.append(score_area(area_name, meta, area_wx, open_issues))

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

    # Summary string
    rain      = weather['curr_rain']
    temp      = weather['curr_temp']
    cond      = weather['current_condition']
    gust      = weather.get('curr_gust', 0)
    fcst      = weather.get('forecast_summary', '')
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
            'summary':            summary,
            'curr_rain':          weather['curr_rain'],
            'curr_temp':          weather['curr_temp'],
            'curr_wind':          weather.get('curr_wind', 0),
            'current_condition':  weather['current_condition'],
            'rain_next_1h':       weather['rain_next_1h'],
            'rain_next_3h':       weather['rain_next_3h'],
            'rain_next_6h':       weather['rain_next_6h'],
            'rain_24h':           weather['rain_24h'],
            'curr_gust':          weather['curr_gust'],
            'curr_humid':         weather['curr_humid'],
            'curr_vis':           weather['curr_vis'],
            'curr_press':         weather['curr_press'],
            'press_trend':        weather['press_trend'],
            'has_thunder':        weather['thunder_now'],
            'thunder_soon':       weather.get('thunder_soon', False),
            'thunder_time':       weather.get('thunder_time'),
            'weather_coming':     weather.get('weather_coming', False),
            'forecast_summary':   weather.get('forecast_summary', ''),
            'forecast_intensity': weather.get('forecast_intensity', 0),
            'max_3h_rain':        weather.get('max_3h_rain', 0),
            'worst_rain_time':    weather.get('worst_rain_time', ''),
            'max_temp_24h':       weather.get('max_temp_24h', temp),
            'max_temp_time':      weather.get('max_temp_time', ''),
            'max_gust_24h':       weather.get('max_gust_24h', gust),
            'storm_now':          weather['storm_now'],
            'peak_rain_hour':     weather['peak_rain_hour'],
            'wind_hazard':        weather.get('wind_hazard', False),
            'heat_hazard':        weather.get('heat_hazard', False),
            'hourly_precip_12h':  weather.get('hourly_precip_12h', []),
            'hourly_codes_12h':   weather.get('hourly_codes_12h', []),
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
        'fetch_lat':       focus_lat if focus_lat is not None else 28.632,
        'fetch_lng':       focus_lng if focus_lng is not None else 77.217,
    }