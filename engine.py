"""
AreaPulse CivicAlert Engine v4.0
==================================
Core logic:
  1. Fetch LIVE weather from Open-Meteo current + minutely_15
  2. Fetch real open issues from Neon Postgres
  3. Run ML model → per-area risk for each civic problem type
  4. Calculate time-to-impact from drainage profile
  5. AI drafts bulletin with actual issue IDs
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
    'Connaught Place':  {'lat':28.6315,'lng':77.2167,'drain':0.00,'elev':0.00,'road_age':1.0,'infra_age':1.0,'wp':0.00,'pop':1.00},
    'Chandni Chowk':    {'lat':28.6507,'lng':77.2334,'drain':0.00,'elev':0.00,'road_age':1.0,'infra_age':1.0,'wp':0.00,'pop':1.00},
    'Paharganj':        {'lat':28.6448,'lng':77.2167,'drain':0.00,'elev':0.00,'road_age':1.0,'infra_age':1.0,'wp':0.00,'pop':1.00},
    'Kashmere Gate':    {'lat':28.6670,'lng':77.2290,'drain':0.00,'elev':0.00,'road_age':1.0,'infra_age':1.0,'wp':0.25,'pop':0.75},
    'Karol Bagh':       {'lat':28.6520,'lng':77.1904,'drain':0.25,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.75},
    'Rohini':           {'lat':28.7493,'lng':77.1000,'drain':0.50,'elev':0.50,'road_age':0.5,'infra_age':0.5,'wp':1.00,'pop':0.50},
    'Dwarka':           {'lat':28.5921,'lng':77.0460,'drain':0.25,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.50},
    'Janakpuri':        {'lat':28.6270,'lng':77.0830,'drain':0.50,'elev':0.50,'road_age':0.5,'infra_age':0.5,'wp':1.00,'pop':0.50},
    'Shahdara':         {'lat':28.6700,'lng':77.2880,'drain':0.00,'elev':0.00,'road_age':1.0,'infra_age':1.0,'wp':0.25,'pop':0.75},
    'Laxmi Nagar':      {'lat':28.6320,'lng':77.2780,'drain':0.25,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.75},
    'Mayur Vihar':      {'lat':28.6090,'lng':77.2970,'drain':0.25,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.50},
    'Saket':            {'lat':28.5245,'lng':77.2066,'drain':1.00,'elev':0.50,'road_age':0.0,'infra_age':0.0,'wp':1.00,'pop':0.50},
    'Hauz Khas':        {'lat':28.5494,'lng':77.2001,'drain':1.00,'elev':0.50,'road_age':0.0,'infra_age':0.0,'wp':1.00,'pop':0.25},
    'Vasant Vihar':     {'lat':28.5570,'lng':77.1570,'drain':1.00,'elev':1.00,'road_age':0.0,'infra_age':0.0,'wp':1.00,'pop':0.25},
    'Defence Colony':   {'lat':28.5710,'lng':77.2300,'drain':1.00,'elev':0.50,'road_age':0.0,'infra_age':0.0,'wp':1.00,'pop':0.25},
    'Pitampura':        {'lat':28.7002,'lng':77.1310,'drain':0.50,'elev':0.50,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.50},
    'Mukherjee Nagar':  {'lat':28.7050,'lng':77.2050,'drain':0.25,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.25,'pop':0.75},
    'Okhla':            {'lat':28.5355,'lng':77.2728,'drain':0.25,'elev':0.25,'road_age':1.0,'infra_age':1.0,'wp':0.25,'pop':0.50},
    'Nizamuddin':       {'lat':28.5892,'lng':77.2469,'drain':0.25,'elev':0.25,'road_age':1.0,'infra_age':1.0,'wp':0.25,'pop':0.50},
    'Lajpat Nagar':     {'lat':28.5700,'lng':77.2373,'drain':0.50,'elev':0.25,'road_age':0.5,'infra_age':0.5,'wp':0.50,'pop':0.75},
}

FLOOD_TIMING_MINS = {0.00:15, 0.25:40, 0.50:100, 1.00:999}

LABELS = ['label_flood','label_pothole_worsen','label_sewage_overflow',
          'label_garbage_flood','label_elec_hazard']

LABEL_DISPLAY = {
    'label_flood':           {'icon':'🌊','name':'Waterlogging',      'dept':'DJB + PWD',        'color':'#1565c0'},
    'label_pothole_worsen':  {'icon':'🕳','name':'Pothole Damage',    'dept':'MCD Roads + PWD',  'color':'#bf360c'},
    'label_sewage_overflow': {'icon':'🚨','name':'Sewage Overflow',   'dept':'DJB',              'color':'#6a1b9a'},
    'label_garbage_flood':   {'icon':'🗑','name':'Garbage Flooding',  'dept':'MCD Sanitation',   'color':'#2e7d32'},
    'label_elec_hazard':     {'icon':'⚡','name':'Electrical Hazard', 'dept':'DISCOM',           'color':'#e65100'},
}

# ── MODEL ─────────────────────────────────────────────────────
_model=None; _encoder=None; _features=None; _loaded=False

def _load_model():
    global _model,_encoder,_features,_loaded
    if _loaded: return True
    d=os.path.join(os.path.dirname(os.path.abspath(__file__)),'models')
    mp=os.path.join(d,'storm_model.pkl')
    ep=os.path.join(d,'area_encoder.pkl')
    mm=os.path.join(d,'model_meta.json')
    if not os.path.exists(mp): return False
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _model=joblib.load(mp); _encoder=joblib.load(ep)
        _features=json.load(open(mm))['features']
        _loaded=True
        print(f"[engine] Storm model loaded ({len(_features)} features)")
        return True
    except Exception as e:
        print(f"[engine] Model load failed: {e}")
        return False

# ── WEATHER ───────────────────────────────────────────────────
def fetch_live_weather(lat, lng):
    """Fetch with separate connect(3s) + read(5s) timeout. Fails fast."""
    try:
        import requests as _req
        r = _req.get(
            'https://api.open-meteo.com/v1/forecast',
            params={
                'latitude':     lat,
                'longitude':    lng,
                'current':      'precipitation,weathercode,temperature_2m,windspeed_10m,windgusts_10m,relativehumidity_2m,surface_pressure,visibility',
                'hourly':       'precipitation,weathercode,temperature_2m,windspeed_10m,surface_pressure,visibility',
                'forecast_days': 1,
                'timezone':     'Asia/Kolkata',
            },
            timeout=(3, 5),   # connect=3s, read=5s — never hangs
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[weather] fetch failed: {e}")
        return None

def parse_live_weather(data):
    if data is None: return None
    c=data.get('current',{})
    curr_rain=float(c.get('precipitation') or 0)
    curr_code=int(c.get('weathercode') or 0)
    curr_temp=float(c.get('temperature_2m') or 25)
    curr_gust=float(c.get('windgusts_10m') or 0)
    curr_humid=float(c.get('relativehumidity_2m') or 60)
    curr_press=float(c.get('surface_pressure') or 1010)
    curr_vis=float(c.get('visibility') or 9999)

    # Parse hourly data FIRST, then use it
    h=data.get('hourly',{})
    def s(k,d=0): return [float(x or d) for x in h.get(k,[])]
    hp=s('precipitation'); hpr=s('surface_pressure',1010)
    hc=[int(x or 0) for x in h.get('weathercode',[])]
    rain_24h=sum(hp[:24]) if hp else 0
    press_trend=(hpr[0]-hpr[2]) if len(hpr)>=3 else 0

    # Use hourly data for 1h and 6h forecasts
    rain_1h=sum(hp[:1]) if hp else curr_rain
    rain_6h=sum(hp[:6]) if hp else curr_rain*6

    WMO={0:'Clear',1:'Mainly clear',2:'Partly cloudy',3:'Overcast',
         45:'Fog',51:'Light drizzle',61:'Light rain',63:'Moderate rain',
         65:'Heavy rain',80:'Rain showers',82:'Violent showers',
         95:'Thunderstorm',96:'Thunderstorm + hail',99:'Severe thunderstorm'}

    # Raw weather flags — passed directly to model as features
    # Model learned from real data what these mean for civic risk
    thunder_now = curr_code >= 95
    rain_now    = curr_code >= 51
    fog_now     = curr_code in [45, 48]
    wind_hazard = curr_gust >= 40
    heat_hazard = curr_temp >= 40

    # storm_now = any non-clear condition — used only for UI mode switching
    storm_now = curr_code > 0 and curr_code not in [0, 1, 2, 3]
    # Also include wind/heat as they affect civic infrastructure
    storm_now = storm_now or wind_hazard or heat_hazard or fog_now

    # weather_intensity — purely for UI display, NOT used in scoring
    # Model scores using raw feature values, not this multiplier
    weather_intensity = min(1.0, (
        (curr_code / 99) * 0.5 +
        (max(0, curr_rain) / 20) * 0.3 +
        (max(0, curr_gust - 20) / 80) * 0.1 +
        (max(0, curr_temp - 30) / 20) * 0.1
    ))
    peak_i=hp[:24].index(max(hp[:24])) if hp else 0
    peak_hr='NOW' if storm_now else (datetime.now()+timedelta(hours=peak_i)).strftime('%I:%M %p')

    # ── FORECAST WINDOWS ─────────────────────────────────────
    # Find worst upcoming weather window in next 24h
    # Used to show "tomorrow" predictions even when clear now

    # Max rain in any 3h window in next 24h
    max_3h_rain = max((sum(hp[i:i+3]) for i in range(0,21)), default=0) if len(hp)>=3 else 0
    # Hour when worst rain starts
    worst_3h_start = max(range(0,21), key=lambda i: sum(hp[i:i+3]), default=0) if len(hp)>=21 else 0
    worst_rain_time = (datetime.now()+timedelta(hours=worst_3h_start)).strftime('%I:%M %p')
    _wrd = datetime.now()+timedelta(hours=worst_3h_start)
    worst_rain_day  = _wrd.strftime('%a %d %b').replace(' 0',' ') if worst_3h_start > 2 else 'Today'

    # Max temp in next 24h
    ht = h.get('temperature_2m',[])
    max_temp_24h = max([float(x or 0) for x in ht[:24]]) if ht else curr_temp
    max_temp_hour = ht[:24].index(max(ht[:24])) if ht and len(ht)>=24 else 0
    max_temp_time = (datetime.now()+timedelta(hours=max_temp_hour)).strftime('%I:%M %p')

    # Max gust in next 24h
    hg = h.get('windgusts_10m',[])
    max_gust_24h = max([float(x or 0) for x in hg[:24]]) if hg else curr_gust

    # Thunder in next 24h
    thunder_soon = any(c>=95 for c in hc[:24])
    thunder_hour = next((i for i,c in enumerate(hc[:24]) if c>=95), None)
    thunder_time = (datetime.now()+timedelta(hours=thunder_hour)).strftime('%I:%M %p') if thunder_hour is not None else None

    # Forecast weather intensity (for scoring future risk)
    if thunder_soon:                        forecast_intensity = 1.0
    elif max_3h_rain > 15:                  forecast_intensity = 0.9
    elif max_3h_rain > 8:                   forecast_intensity = 0.75
    elif max_3h_rain > 2:                   forecast_intensity = 0.5
    elif max_temp_24h >= 44:                forecast_intensity = 0.8
    elif max_temp_24h >= 40:               forecast_intensity = 0.6
    elif max_gust_24h >= 60:               forecast_intensity = 0.7
    elif max_gust_24h >= 40:               forecast_intensity = 0.5
    else:                                   forecast_intensity = 0.2

    # Is there ANY meaningful weather coming in next 24h?
    weather_coming = (
        max_3h_rain > 2 or thunder_soon or
        max_temp_24h >= 40 or max_gust_24h >= 40
    )

    # Build forecast summary string
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
        'curr_rain':round(curr_rain,1),'curr_code':curr_code,
        'curr_temp':round(curr_temp,1),'curr_gust':round(curr_gust,1),
        'curr_humid':round(curr_humid,1),'curr_press':round(curr_press,1),
        'curr_vis':round(curr_vis,0),'press_trend':round(press_trend,2),
        'current_condition':WMO.get(curr_code,f'Code {curr_code}'),
        'rain_next_1h':round(rain_1h,1),'rain_next_6h':round(rain_6h,1),
        'rain_24h':round(rain_24h,1),

        # Storm flags
        'storm_now':storm_now,'thunder_now':thunder_now,'rain_now':rain_now,
        'wind_hazard':wind_hazard,'heat_hazard':heat_hazard,'fog_now':fog_now,
        'weather_intensity':weather_intensity,
        'thunder_soon':thunder_soon,'thunder_time':thunder_time,
        'fog':curr_vis<500,'dense_fog':curr_vis<200,
        'peak_rain_hour':peak_hr,

        # 24h forecast
        'max_3h_rain':round(max_3h_rain,1),
        'worst_rain_time':worst_rain_time,
        'worst_rain_day':worst_rain_day,
        'max_temp_24h':round(max_temp_24h,1),
        'max_temp_time':max_temp_time,
        'max_gust_24h':round(max_gust_24h,1),
        'weather_coming':weather_coming,
        'forecast_intensity':forecast_intensity,
        'forecast_summary':forecast_summary,

        'month':datetime.now().month,'hour':datetime.now().hour,
    }

def fetch_aqi(token='demo'):
    # Try AQICN with real token
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
            print(f"[aqi] AQICN error: {d.get('data','unknown')}")
    except Exception as e:
        print(f"[aqi] fetch failed: {e}")

    # Fallback: try open-meteo UV / European AQI (no key needed)
    try:
        import requests as _req
        r = _req.get(
            'https://air-quality-api.open-meteo.com/v1/air-quality',
            params={
                'latitude': 28.65, 'longitude': 77.22,
                'current': 'us_aqi,pm2_5,pm10',
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
    area_issues=[i for i in open_issues
                 if str(i.get('area','')).strip().lower()==area_name.strip().lower()
                 and i.get('status') not in ('resolved',)]
    w_open=sum(1 for i in area_issues if i.get('tag')=='water')
    s_open=sum(1 for i in area_issues if i.get('tag')=='sewage')
    p_open=sum(1 for i in area_issues if i.get('tag')=='pothole')
    g_open=sum(1 for i in area_issues if i.get('tag')=='garbage')
    e_open=sum(1 for i in area_issues if i.get('tag')=='electricity')
    now_ts=time.time()
    complaint_vel=sum(1 for i in area_issues if now_ts-float(i.get('timestamp') or 0)<7200)

    # If nothing happening NOW but weather is coming in 24h,
    # score using the FORECAST worst window so dashboard shows future risk
    using_forecast = (not weather.get('storm_now', False) and
                      weather.get('weather_coming', False))

    eff_rain    = weather.get('max_3h_rain', 0)     if using_forecast else weather['curr_rain']
    eff_temp    = weather.get('max_temp_24h', weather['curr_temp']) if using_forecast else weather['curr_temp']
    eff_gust    = weather.get('max_gust_24h', weather['curr_gust']) if using_forecast else weather['curr_gust']
    eff_thunder = weather.get('thunder_soon', False)  if using_forecast else weather.get('thunder_now', False)
    eff_wi      = weather.get('forecast_intensity', 0.2) if using_forecast else weather.get('weather_intensity', 0.3)
    eff_code    = (63 if eff_rain > 5 else 51) if using_forecast else weather['curr_code']

    scores={}
    if _load_model():
        try:
            try: ae=int(_encoder.transform([area_name])[0])
            except: ae=0
            row={'area_enc':ae,'drain':meta['drain'],'elev':meta['elev'],
                 'road_age':meta['road_age'],'infra_age':meta['infra_age'],
                 'wp':meta['wp'],'pop':meta['pop'],
                 'month':weather['month'],'hour':weather['hour'],
                 'rain_1h':eff_rain,'rain_3h':weather['rain_next_1h'],
                 'rain_6h':weather['rain_next_6h'],'rain_24h':weather['rain_24h'],
                 'temp':eff_temp,'wind':0,'gust':eff_gust,
                 'humid':weather['curr_humid'],'pressure':weather['curr_press'],
                 'press_trend':weather['press_trend'],'visibility':weather['curr_vis'],
                 'weathercode':eff_code,
                 'storm_now':int(weather.get('storm_now',False) or using_forecast),
                 'thunder_now':int(eff_thunder),
                 'open_water':w_open,'open_sewage':s_open,'open_pothole':p_open,
                 'open_garbage':g_open,'open_elec':e_open,'complaint_vel':complaint_vel}
            fd=pd.DataFrame([row])[_features]
            probs=np.array(_model.predict_proba(fd))
            rule_scores=_rules(meta,weather,w_open,s_open,p_open,g_open,e_open,
                              eff_rain=eff_rain,eff_temp=eff_temp,
                              eff_gust=eff_gust,eff_wi=eff_wi,eff_thunder=eff_thunder)
            for i,label in enumerate(LABELS):
                vuln=_vuln(meta,label)
                ml_score=int(probs[i][0][1]*100*vuln)
                rule_score=rule_scores.get(label,0)
                # Take max of ML and rules — if ML was trained on limited data
                # rules provide a floor so scores are never 0 when weather exists
                scores[label]=min(max(ml_score, rule_score),99)
        except Exception as e:
            print(f"[model] {e}")
            scores=_rules(meta,weather,w_open,s_open,p_open,g_open,e_open,
                          eff_rain=eff_rain,eff_temp=eff_temp,
                          eff_gust=eff_gust,eff_wi=eff_wi,eff_thunder=eff_thunder)
    else:
        scores=_rules(meta,weather,w_open,s_open,p_open,g_open,e_open,
                      eff_rain=eff_rain,eff_temp=eff_temp,
                      eff_gust=eff_gust,eff_wi=eff_wi,eff_thunder=eff_thunder)

    rain=weather['curr_rain']
    dk=min(FLOOD_TIMING_MINS.keys(),key=lambda x:abs(x-meta['drain']))
    base=FLOOD_TIMING_MINS[dk]
    tf=0.4 if rain>20 else 0.6 if rain>10 else 0.8 if rain>5 else 1.0
    isf=max(0.5,1.0-(s_open+w_open)*0.1)
    tti=int(base*tf*isf)

    tag_risk={'pothole':'label_pothole_worsen','sewage':'label_sewage_overflow',
              'water':'label_flood','garbage':'label_garbage_flood',
              'electricity':'label_elec_hazard','streetlight':'label_elec_hazard'}
    affected=[]
    for issue in area_issues:
        tag=issue.get('tag',''); risk=tag_risk.get(tag)
        if risk and scores.get(risk,0)>=30:
            affected.append({'id':issue.get('id'),'tag':tag,
                             'severity':issue.get('severity','medium'),
                             'description':str(issue.get('description',''))[:60],
                             'risk_label':risk,'risk_score':scores.get(risk,0)})

    overall=max(scores.values()) if scores else 0
    return {
        'area':area_name,'lat':meta['lat'],'lng':meta['lng'],
        'scores':scores,'overall_risk':overall,
        'risk_level':'CRITICAL' if overall>=75 else 'HIGH' if overall>=50 else 'MEDIUM' if overall>=25 else 'LOW',
        'time_to_impact_mins':tti,
        'affected_issues':sorted(affected,key=lambda x:x['risk_score'],reverse=True),
        'open_issues_count':len(area_issues),
        'complaint_velocity':complaint_vel,
        'storm_amplified':(weather.get('storm_now',False) or weather.get('weather_coming',False)) and overall>0,
        'drain_quality':('very poor' if meta['drain']<=0 else 'poor' if meta['drain']<=0.25 else 'medium' if meta['drain']<=0.5 else 'good'),
    }

def _vuln(meta,label):
    if label=='label_flood': return 0.4+(1-meta['drain'])*0.4+(1-meta['elev'])*0.2
    if label=='label_pothole_worsen': return 0.3+meta['road_age']*0.5+(1-meta['drain'])*0.2
    if label=='label_sewage_overflow': return 0.4+(1-meta['drain'])*0.4+meta['pop']*0.2
    if label=='label_garbage_flood': return 0.4+meta['pop']*0.3+(1-meta['drain'])*0.3
    if label=='label_elec_hazard': return 0.5+meta['infra_age']*0.3+meta['pop']*0.2
    return 0.6

def _rules(meta, weather, w, s, p, g, e,
           eff_rain=None, eff_temp=None, eff_gust=None,
           eff_wi=None, eff_thunder=None):
    # Fallback when ML model not available.
    # Uses raw weather values directly — no hardcoded thresholds.
    rain    = eff_rain if eff_rain is not None else weather.get('curr_rain', 0)
    temp    = eff_temp if eff_temp is not None else weather.get('curr_temp', 25)
    gust    = eff_gust if eff_gust is not None else weather.get('curr_gust', 0)
    code    = weather.get('curr_code', 0)
    press_d = abs(weather.get('press_trend', 0))
    humid   = weather.get('curr_humid', 60)
    dv      = meta['drain']
    rv      = meta['road_age']
    iv      = meta['infra_age']
    pv      = meta['pop']
    def cl(x): return min(int(max(x, 0)), 99)
    return {
        'label_flood': cl(
            rain*3*(1-dv) + s*8*(1-dv) + w*6*(1-dv) + press_d*2*(1-dv)
        ),
        'label_pothole_worsen': cl(
            rain*2*rv + temp*0.3*rv + gust*0.2*rv + p*10*rv
        ),
        'label_sewage_overflow': cl(
            rain*4*(1-dv) + s*12*(1-dv) + humid*0.1*(1-dv)
        ),
        'label_garbage_flood': cl(
            rain*1.5 + gust*0.3 + temp*0.2 + g*8
        ),
        'label_elec_hazard': cl(
            (30 if code>=95 else 0) + gust*0.5*iv + temp*0.4*iv + e*12*iv + pv*5
        ),
    }

def generate_bulletin(areas, weather, aqi, groq_api_key=None):
    storm=weather.get('storm_now',False)
    top=[a for a in areas[:4] if a['overall_risk']>=25]
    if not top: return "No significant civic risk. Routine monitoring in effect."

    ctx=[]
    for a in top[:3]:
        ids=[f"#AP-{i['id']}" for i in a['affected_issues'][:3] if i.get('id')]
        tr=max(a['scores'],key=a['scores'].get) if a['scores'] else 'label_flood'
        ctx.append(f"{a['area']}: {a['overall_risk']}/100, ~{a['time_to_impact_mins']}min, "
                   f"{LABEL_DISPLAY.get(tr,{}).get('name',tr)}, issues: {', '.join(ids) or 'none'}")

    pfx="ACTIVE STORM — " if storm else ""
    prompt=(f"You are AreaPulse civic AI for Delhi officers.\n\n"
            f"{pfx}Weather: {weather.get('current_condition','—')} · "
            f"Rain now: {weather.get('curr_rain',0)}mm/hr · "
            f"Next 6h: {weather.get('rain_next_6h',0)}mm\n\n"
            f"At-risk areas:\n"+"\n".join(ctx)+
            f"\n\nWrite 3 sentences: (1) severity now, (2) top 2 areas with issue IDs + dept, "
            f"(3) specific action with time deadline. Name issue IDs. Name departments. Max 90 words.")

    if groq_api_key:
        try:
            from groq import Groq
            resp=Groq(api_key=groq_api_key).chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{'role':'user','content':prompt}],
                max_tokens=160,temperature=0.3)
            return resp.choices[0].message.content
        except Exception as e: print(f"[groq] {e}")

    t=top[0]; tr=max(t['scores'],key=t['scores'].get) if t['scores'] else 'label_flood'
    ids=[f"#AP-{i['id']}" for i in t['affected_issues'][:2] if i.get('id')]
    return (f"{'⛈ STORM ACTIVE — ' if storm else ''}"
            f"{weather.get('current_condition','Weather alert')} in Delhi. "
            f"Highest risk: {t['area']} ({t['overall_risk']}/100) — "
            f"{LABEL_DISPLAY.get(tr,{}).get('name','flooding')} in ~{t['time_to_impact_mins']}min. "
            f"{'Issues '+', '.join(ids)+' will worsen. ' if ids else ''}"
            f"Deploy {LABEL_DISPLAY.get(tr,{}).get('dept','relevant dept')} immediately.")

def run_full_prediction(open_issues=None, groq_api_key=None, aqi_token='demo'):
    open_issues=open_issues or []; _load_model()
    print(f"[engine] v4 · {len(DELHI_AREAS)} areas · {len(open_issues)} issues")

    print("[engine] Fetching live weather...")
    # Single fetch for Delhi center — fast, covers all 20 areas
    wx_n=fetch_live_weather(28.65,77.22)
    if wx_n is None:
        raise Exception("Open-Meteo weather API unreachable. Check your internet connection.")
    wn=parse_live_weather(wx_n)
    ws=wn  # same weather object for south Delhi areas

    print("[engine] Fetching AQI...")
    aqi=fetch_aqi(aqi_token)

    results=[]
    for area_name,meta in DELHI_AREAS.items():
        wx=wn if meta['lat']>28.63 else ws
        results.append(score_area(area_name,meta,wx,open_issues))

    w=wn
    storm_mode = w.get('storm_now',False) or w.get('weather_coming',False)
    if storm_mode:
        results.sort(key=lambda x:(-x['overall_risk'],x['time_to_impact_mins']))
    else:
        results.sort(key=lambda x:x['overall_risk'],reverse=True)

    rain=w['curr_rain']; temp=w['curr_temp']; cond=w['current_condition']
    gust=w.get('curr_gust',0)
    fcst=w.get('forecast_summary','')

    # Summary: current conditions first, then forecast if clear now
    if w.get('thunder_now'):
        summary=f'⛈ THUNDERSTORM NOW — {cond} · {rain}mm/hr · gusts {gust}km/h'
    elif w.get('rain_now') and rain>10:
        summary=f'🌧 HEAVY RAIN NOW — {cond} · {rain}mm/hr · {w["rain_next_6h"]}mm next 6h'
    elif w.get('rain_now') and rain>0:
        summary=f'🌦 RAIN NOW — {cond} · {rain}mm/hr'
    elif w.get('wind_hazard') and gust>60:
        summary=f'💨 STRONG WINDS NOW — gusts {gust}km/h · {cond}'
    elif w.get('wind_hazard'):
        summary=f'🌬 WINDY NOW — gusts {gust}km/h · {cond}'
    elif w.get('heat_hazard') and temp>44:
        summary=f'🌡 EXTREME HEAT NOW — {temp}°C · {cond}'
    elif w.get('heat_hazard'):
        summary=f'🔆 HEATWAVE NOW — {temp}°C · {cond}'
    elif w.get('fog_now'):
        summary=f'🌫 FOG NOW — {int(w["curr_vis"])}m visibility'
    # Nothing now — show what's coming
    elif w.get('weather_coming'):
        summary=f'☀ Clear now · Forecast: {fcst}'
    elif aqi and aqi['aqi']>200:
        summary=f'😷 Poor air quality — AQI {aqi["aqi"]}'
    elif temp<6:
        summary=f'❄ Cold wave — {temp}°C · {cond}'
    else:
        summary=f'☀ {cond} — {temp}°C · No weather alerts'

    bulletin=generate_bulletin(results,w,aqi,groq_api_key)

    label_summary={}
    for label in LABELS:
        vals=[r['scores'].get(label,0) for r in results]
        label_summary[label]={
            'avg':round(sum(vals)/max(len(vals),1)),
            'max':max(vals),
            'critical_areas':[r['area'] for r in results if r['scores'].get(label,0)>=75],
            'display':LABEL_DISPLAY.get(label,{}),
        }

    return {
        'areas':results,'weather':{
            'summary':summary,'curr_rain':w['curr_rain'],
            'curr_temp':w['curr_temp'],'current_condition':w['current_condition'],
            'rain_next_1h':w['rain_next_1h'],'rain_next_6h':w['rain_next_6h'],
            'rain_24h':w['rain_24h'],'curr_gust':w['curr_gust'],
            'curr_humid':w['curr_humid'],'curr_vis':w['curr_vis'],
            'curr_press':w['curr_press'],'press_trend':w['press_trend'],
            'has_thunder':w['thunder_now'],'thunder_soon':w.get('thunder_soon',False),
            'thunder_time':w.get('thunder_time'),
            'weather_coming':w.get('weather_coming',False),
            'forecast_summary':w.get('forecast_summary',''),
            'forecast_intensity':w.get('forecast_intensity',0),
            'max_3h_rain':w.get('max_3h_rain',0),
            'worst_rain_time':w.get('worst_rain_time',''),
            'max_temp_24h':w.get('max_temp_24h',temp),
            'max_gust_24h':w.get('max_gust_24h',gust),
            'storm_now':w['storm_now'],'peak_rain_hour':w['peak_rain_hour'],
        },
        'aqi':aqi,'ai_bulletin':bulletin,'storm_mode':storm_mode,
        'label_summary':label_summary,
        'total_at_risk':sum(1 for r in results if r['overall_risk']>=50),
        'critical_areas':[r['area'] for r in results if r['overall_risk']>=75],
        'generated_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'issues_analysed':len(open_issues),
        'ml_active':_load_model(),
        'label_display':LABEL_DISPLAY,
    }