"""
STEP 1 — Download Real Historical Weather Data
================================================
Run on YOUR machine:  python download_training_data.py

Downloads real hourly weather for all 20 Delhi areas
from Open-Meteo historical API (free, no key needed).

Covers 6 monsoon seasons (2019-2024) = ~600 storm days per area.
Takes ~25 minutes due to rate limiting. Resumes if interrupted.

Output: data/weather_history.csv
"""

import requests
import pandas as pd
import json
import time
import os
from datetime import datetime

DELHI_AREAS = {
    'Connaught Place':  (28.6315, 77.2167, 'very_poor', 'very_low',  'old',    'old',    'very_low',  'very_high'),
    'Chandni Chowk':    (28.6507, 77.2334, 'very_poor', 'very_low',  'old',    'old',    'very_low',  'very_high'),
    'Paharganj':        (28.6448, 77.2167, 'very_poor', 'very_low',  'old',    'old',    'very_low',  'very_high'),
    'Kashmere Gate':    (28.6670, 77.2290, 'very_poor', 'very_low',  'old',    'old',    'low',       'high'),
    'Karol Bagh':       (28.6520, 77.1904, 'poor',      'low',       'medium', 'medium', 'medium',    'high'),
    'Rohini':           (28.7493, 77.1000, 'medium',    'medium',    'medium', 'medium', 'good',      'medium'),
    'Dwarka':           (28.5921, 77.0460, 'poor',      'low',       'medium', 'medium', 'medium',    'medium'),
    'Janakpuri':        (28.6270, 77.0830, 'medium',    'medium',    'medium', 'medium', 'good',      'medium'),
    'Shahdara':         (28.6700, 77.2880, 'very_poor', 'very_low',  'old',    'old',    'low',       'high'),
    'Laxmi Nagar':      (28.6320, 77.2780, 'poor',      'low',       'medium', 'medium', 'medium',    'high'),
    'Mayur Vihar':      (28.6090, 77.2970, 'poor',      'low',       'medium', 'medium', 'medium',    'medium'),
    'Saket':            (28.5245, 77.2066, 'good',      'medium',    'new',    'new',    'good',      'medium'),
    'Hauz Khas':        (28.5494, 77.2001, 'good',      'medium',    'new',    'new',    'good',      'low'),
    'Vasant Vihar':     (28.5570, 77.1570, 'good',      'high',      'new',    'new',    'good',      'low'),
    'Defence Colony':   (28.5710, 77.2300, 'good',      'medium',    'new',    'new',    'good',      'low'),
    'Pitampura':        (28.7002, 77.1310, 'medium',    'medium',    'medium', 'medium', 'medium',    'medium'),
    'Mukherjee Nagar':  (28.7050, 77.2050, 'poor',      'low',       'medium', 'medium', 'low',       'high'),
    'Okhla':            (28.5355, 77.2728, 'poor',      'low',       'old',    'old',    'low',       'medium'),
    'Nizamuddin':       (28.5892, 77.2469, 'poor',      'low',       'old',    'old',    'low',       'medium'),
    'Lajpat Nagar':     (28.5700, 77.2373, 'medium',    'low',       'medium', 'medium', 'medium',    'high'),
}

ENC = {
    'drainage':   {'very_poor':0.0,'poor':0.25,'medium':0.5,'good':1.0},
    'elevation':  {'very_low':0.0,'low':0.25,'medium':0.5,'high':1.0},
    'road_age':   {'old':1.0,'medium':0.5,'new':0.0},
    'water_pres': {'very_low':0.0,'low':0.25,'medium':0.5,'good':1.0},
    'pop_den':    {'very_high':1.0,'high':0.75,'medium':0.5,'low':0.25},
}

# Full monsoon seasons + pre-monsoon (heat risk)
SEASONS = [
    ('2019-04-01', '2019-10-31'),
    ('2020-04-01', '2020-10-31'),
    ('2021-04-01', '2021-10-31'),
    ('2022-04-01', '2022-10-31'),
    ('2023-04-01', '2023-10-31'),
    ('2024-04-01', '2024-10-31'),
]

def fetch_historical(lat, lng, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude':   lat,
        'longitude':  lng,
        'start_date': start,
        'end_date':   end,
        'hourly': ','.join([
            'precipitation', 'weathercode',
            'temperature_2m', 'windspeed_10m', 'windgusts_10m',
            'relativehumidity_2m', 'surface_pressure', 'visibility',
        ]),
        'timezone': 'Asia/Kolkata',
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def build_rows(area, lat, lng, drain, elev, road, infra, wp, pop, weather_data):
    rows = []
    h       = weather_data.get('hourly', {})
    times   = h.get('time', [])
    precip  = [float(x or 0) for x in h.get('precipitation', [])]
    codes   = [int(x or 0)   for x in h.get('weathercode', [])]
    temps   = [float(x or 25) for x in h.get('temperature_2m', [])]
    winds   = [float(x or 0)  for x in h.get('windspeed_10m', [])]
    gusts   = [float(x or 0)  for x in h.get('windgusts_10m', [])]
    humids  = [float(x or 60) for x in h.get('relativehumidity_2m', [])]
    press   = [float(x or 1010) for x in h.get('surface_pressure', [])]
    vis     = [float(x or 9999) for x in h.get('visibility', [])]

    drain_v = ENC['drainage'][drain]
    elev_v  = ENC['elevation'][elev]
    road_v  = ENC['road_age'][road]
    infra_v = ENC['road_age'][infra]
    wp_v    = ENC['water_pres'][wp]
    pop_v   = ENC['pop_den'][pop]

    for i in range(len(times)):
        try:
            dt = datetime.fromisoformat(times[i])
        except:
            continue

        # Rolling features — what happened in last N hours
        rain_1h  = precip[i]
        rain_3h  = sum(precip[max(0,i-2):i+1])
        rain_6h  = sum(precip[max(0,i-5):i+1])
        rain_24h = sum(precip[max(0,i-23):i+1])

        # Pressure trend (falling pressure = incoming storm)
        press_trend = 0
        if i >= 3:
            press_trend = press[i] - press[i-3]  # negative = falling

        code     = codes[i]
        temp     = temps[i]
        wind     = winds[i]
        gust     = gusts[i]
        humid    = humids[i]
        min_vis  = vis[i]
        month    = dt.month
        hour     = dt.hour

        # Is a storm happening RIGHT NOW at this hour?
        storm_now    = int(code >= 61)   # any rain/storm
        thunder_now  = int(code >= 95)   # thunderstorm

        rows.append({
            'area':        area,
            'datetime':    times[i],
            'month':       month,
            'hour':        hour,

            # Area fixed features
            'drain':       drain_v,
            'elev':        elev_v,
            'road_age':    road_v,
            'infra_age':   infra_v,
            'wp':          wp_v,
            'pop':         pop_v,

            # Live weather features
            'rain_1h':     round(rain_1h, 2),
            'rain_3h':     round(rain_3h, 2),
            'rain_6h':     round(rain_6h, 2),
            'rain_24h':    round(rain_24h, 2),
            'temp':        round(temp, 1),
            'wind':        round(wind, 1),
            'gust':        round(gust, 1),
            'humid':       round(humid, 1),
            'pressure':    round(press[i], 1),
            'press_trend': round(press_trend, 2),
            'visibility':  round(min_vis, 0),
            'weathercode': code,
            'storm_now':   storm_now,
            'thunder_now': thunder_now,
        })
    return rows

def main():
    os.makedirs('data', exist_ok=True)
    cache_path = 'data/weather_cache.json'
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    all_rows = []
    total    = len(DELHI_AREAS) * len(SEASONS)
    done     = 0

    for area, (lat, lng, drain, elev, road, infra, wp, pop) in DELHI_AREAS.items():
        for start, end in SEASONS:
            key = f"{area}|{start}"
            if key not in cache:
                print(f"Downloading {area} {start[:4]}...", end=' ', flush=True)
                try:
                    data = fetch_historical(lat, lng, start, end)
                    cache[key] = data
                    with open(cache_path, 'w') as f:
                        json.dump(cache, f)
                    print(f"OK ({len(data.get('hourly',{}).get('time',[]))} hours)")
                    time.sleep(0.5)
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

            rows = build_rows(area, lat, lng, drain, elev, road, infra, wp, pop, cache[key])
            all_rows.extend(rows)
            done += 1
            if done % 5 == 0:
                print(f"  {done}/{total} complete ({done*100//total}%)")

    df = pd.DataFrame(all_rows)
    out = 'data/weather_history.csv'
    df.to_csv(out, index=False)

    print(f"\n{'='*50}")
    print(f"Downloaded {len(df):,} hourly records")
    print(f"Storm hours (code>=61): {df['storm_now'].sum():,}")
    print(f"Thunder hours (code>=95): {df['thunder_now'].sum():,}")
    print(f"Saved to {out}")

if __name__ == '__main__':
    main()
