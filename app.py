"""
AreaPulse CivicAlert — Standalone App
======================================
Phase 1 + Phase 2 fixes applied:
  [P1] GOV_ACCOUNTS now include area, lat, lng per officer
  [P1] Login stores area/lat/lng in session
  [P1] New /api/predict-for-officer route — fetches weather for officer's exact location
  [P1] Frontend receives officer_area, officer_lat, officer_lng in response
  [P2] Cache TTL raised to 300s (matches main AreaPulse app)
  [P2] engine.py handles all ML fixes (rain_3h, wind, tti, encoder)

Run: python app.py
Open: http://localhost:5050
"""

import os, json, time, threading
from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from dotenv import load_dotenv

load_dotenv()

from engine import (
    run_full_prediction, DELHI_AREAS,
    fetch_aqi, fetch_issues_from_postgres,
)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'civicalert-dev-2026')

GROQ_KEY     = os.environ.get('GROQ_API_KEY', '')
AQI_TOKEN    = os.environ.get('AQICN_TOKEN', 'demo')
DATABASE_URL = os.environ.get('DATABASE_URL', '')

# ── MOCK ISSUES ───────────────────────────────────────────────
MOCK_ISSUES = [
    {'id':1,  'area':'Chandni Chowk',   'tag':'sewage',      'severity':'high',   'status':'open',       'description':'Open manhole on main road',              'timestamp': time.time() - 3600},
    {'id':2,  'area':'Chandni Chowk',   'tag':'water',       'severity':'high',   'status':'open',       'description':'Water pipe burst near market',           'timestamp': time.time() - 7200},
    {'id':3,  'area':'Chandni Chowk',   'tag':'pothole',     'severity':'medium', 'status':'in_progress','description':'Large pothole near metro gate',          'timestamp': time.time() - 1800},
    {'id':4,  'area':'Kashmere Gate',   'tag':'sewage',      'severity':'high',   'status':'escalated',  'description':'Sewage overflow, foul smell',            'timestamp': time.time() - 5400},
    {'id':5,  'area':'Kashmere Gate',   'tag':'streetlight', 'severity':'medium', 'status':'open',       'description':'5 streetlights out on main road',        'timestamp': time.time() - 9000},
    {'id':6,  'area':'Paharganj',       'tag':'water',       'severity':'high',   'status':'open',       'description':'No water for 3 days',                    'timestamp': time.time() - 86400},
    {'id':7,  'area':'Paharganj',       'tag':'garbage',     'severity':'medium', 'status':'open',       'description':'Garbage not collected 2 weeks',          'timestamp': time.time() - 172800},
    {'id':8,  'area':'Rohini',          'tag':'pothole',     'severity':'medium', 'status':'open',       'description':'Pothole near school',                    'timestamp': time.time() - 3600},
    {'id':9,  'area':'Dwarka',          'tag':'electricity', 'severity':'high',   'status':'open',       'description':'Daily power cuts 3–5pm',                 'timestamp': time.time() - 7200},
    {'id':10, 'area':'Dwarka',          'tag':'water',       'severity':'medium', 'status':'open',       'description':'Low water pressure',                     'timestamp': time.time() - 10800},
    {'id':11, 'area':'Saket',           'tag':'tree',        'severity':'low',    'status':'open',       'description':'Tree branches touching power lines',     'timestamp': time.time() - 43200},
    {'id':12, 'area':'Mayur Vihar',     'tag':'sewage',      'severity':'high',   'status':'open',       'description':'Drain blocked, standing water',          'timestamp': time.time() - 3600},
    {'id':13, 'area':'Karol Bagh',      'tag':'traffic',     'severity':'medium', 'status':'open',       'description':'Traffic signal not working',             'timestamp': time.time() - 1800},
    {'id':14, 'area':'Mukherjee Nagar', 'tag':'noise',       'severity':'low',    'status':'open',       'description':'Construction noise after 10pm',          'timestamp': time.time() - 7200},
    {'id':15, 'area':'Okhla',           'tag':'garbage',     'severity':'high',   'status':'escalated',  'description':'Industrial waste dumped illegally',      'timestamp': time.time() - 3600},
]

# ── CACHE ─────────────────────────────────────────────────────
_cache = {'result': None, 'ts': 0}
CACHE_TTL = 120   # 2 min — during rain conditions change fast; 5min was causing stale "clear" data

# ── GOV ACCOUNTS ──────────────────────────────────────────────
# FIX [P1]: each account now carries area name + lat/lng of their jurisdiction
GOV_ACCOUNTS = {
    'gov_rmc': {
        'pin':       '0000',
        'name':      'RMC Officer',
        'authority': 'Ranchi Municipal Corporation',
        'tags':      ['pothole', 'garbage', 'sewage', 'streetlight', 'tree', 'other'],
        'area':      'Connaught Place',
        'lat':        28.6315,
        'lng':        77.2167,
    },
    'gov_water': {
        'pin':       '0000',
        'name':      'Water Board Officer',
        'authority': 'Drinking Water & Sanitation Dept (Jharkhand)',
        'tags':      ['water'],
        'area':      'Karol Bagh',
        'lat':        28.6520,
        'lng':        77.1904,
    },
    'gov_electricity': {
        'pin':       '0000',
        'name':      'Electricity Officer',
        'authority': 'Jharkhand Bijli Vitran Nigam (JBVNL)',
        'tags':      ['electricity'],
        'area':      'Dwarka',
        'lat':        28.5921,
        'lng':        77.0460,
    },
    'gov_traffic': {
        'pin':       '0000',
        'name':      'Traffic Police',
        'authority': 'Ranchi Traffic Police',
        'tags':      ['traffic', 'noise'],
        'area':      'Chandni Chowk',
        'lat':        28.6507,
        'lng':        77.2334,
    },
}


# ── HELPERS ───────────────────────────────────────────────────
def _get_issues():
    """Return issues from Postgres if available, else mock data."""
    if DATABASE_URL:
        try:
            issues = fetch_issues_from_postgres(DATABASE_URL)
            return issues, f'Neon Postgres ({len(issues)} real issues)'
        except Exception as db_err:
            print(f"[db] {db_err} — falling back to mock issues")
    return MOCK_ISSUES, 'mock (DATABASE_URL not set)'


def _run_prediction_thread(issues, focus_lat=None, focus_lng=None, rain_override_mm=0):
    """
    Run run_full_prediction in a thread with a 45s hard timeout.
    Returns (result_dict, error_string).
    """
    result_box = [None]
    error_box  = [None]

    def _run():
        try:
            result_box[0] = run_full_prediction(
                open_issues=issues,
                groq_api_key=GROQ_KEY,
                aqi_token=AQI_TOKEN,
                focus_lat=focus_lat,
                focus_lng=focus_lng,
                rain_override_mm=rain_override_mm,
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[engine] CRASH:\n{tb}")
            error_box[0] = f"{type(e).__name__}: {str(e)}"

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=45)  # 5 zone fetches + AQI + 36 scorings + Groq = up to 20s

    if t.is_alive():
        return None, "Prediction timed out after 45s. Check internet connection."
    if error_box[0]:
        return None, error_box[0]
    return result_box[0], None


# ── ROUTES ────────────────────────────────────────────────────
@app.route('/')
def index():
    gov = session.get('gov_role')
    return render_template('index.html', gov=gov)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip().lower()
        pin      = (request.form.get('pin')      or '').strip()

        gov = GOV_ACCOUNTS.get(username)
        if not gov:
            return render_template('login.html', error='Unknown government account.')
        if pin != gov['pin']:
            return render_template('login.html', error='Incorrect PIN.')

        # FIX [P1]: store area, lat, lng in session so the frontend can use them
        session['gov_role'] = {
            'username':  username,
            'name':      gov['name'],
            'authority': gov['authority'],
            'tags':      gov['tags'],
            'area':      gov['area'],
            'lat':       gov['lat'],
            'lng':       gov['lng'],
        }
        return redirect(url_for('index'))

    if session.get('gov_role'):
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('gov_role', None)
    return redirect(url_for('login'))


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Standard full-Delhi prediction.
    POST body (optional): { "force_refresh": false }
    """
    global _cache

    body  = request.get_json(silent=True) or {}
    force = body.get('force_refresh', False)

    # Serve cache if fresh and not forced
    if not force and _cache['result'] and (time.time() - _cache['ts']) < CACHE_TTL:
        result = dict(_cache['result'])
        result['cached']      = True
        result['cache_age_s'] = int(time.time() - _cache['ts'])
        return jsonify(result)

    issues, issue_source = _get_issues()

    result, err = _run_prediction_thread(issues)

    if err:
        if _cache['result']:
            stale = dict(_cache['result'])
            stale['cached']      = True
            stale['cache_age_s'] = int(time.time() - _cache['ts'])
            stale['warning']     = f'Live fetch failed ({err}) — showing last known data'
            return jsonify(stale)
        return jsonify({'error': err}), 500

    result['cached']       = False
    result['issue_source'] = issue_source
    _cache['result'] = result
    _cache['ts']     = time.time()
    return jsonify(result)


# ── LOCATION-AWARE PREDICTION (used by all visitors) ─────────
@app.route('/api/predict-for-officer', methods=['POST'])
def predict_for_officer():
    """
    Location-aware prediction for any visitor.
    Every user (gov official, NGO, anyone) sends their browser lat/lng.
    Weather is fetched for their exact location so they see what's
    happening where they actually are.

    Falls back to Delhi center (28.61, 77.20) if no location provided.

    POST body:
    {
        "lat":           28.6315,   // browser geolocation lat (optional)
        "lng":           77.2167,   // browser geolocation lng (optional)
        "force_refresh": false
    }
    """
    global _cache

    body  = request.get_json(silent=True) or {}
    force = body.get('force_refresh', False)

    # Prefer session values; POST body can override (for browser geolocation)
    gov = session.get('gov_role', {})
    lat  = float(body.get('lat')  or gov.get('lat')  or 28.6139)
    lng  = float(body.get('lng')  or gov.get('lng')  or 77.2090)
    area = str(body.get('area')   or gov.get('area') or 'Delhi')
    rain_override_mm = int(body.get('rain_override_mm') or 0)
    print(f"[predict] location: lat={lat:.4f} lng={lng:.4f} area={area} "
          f"rain_override={rain_override_mm}mm "
          f"(from: {'body' if body.get('lat') else 'session' if gov.get('lat') else 'default'})")

    # Officer-specific cache key — include override so different rain levels don't share cache
    officer_cache_key = f"{lat:.4f},{lng:.4f}:{rain_override_mm}"
    cache_entry = _cache.get(officer_cache_key)

    # Skip cache when override is active — user wants fresh scoring
    if not force and not rain_override_mm and cache_entry and (time.time() - cache_entry['ts']) < CACHE_TTL:
        result = dict(cache_entry['result'])
        result['cached']         = True
        result['cache_age_s']    = int(time.time() - cache_entry['ts'])
        result['officer_area']   = area
        result['officer_lat']    = lat
        result['officer_lng']    = lng
        return jsonify(result)

    issues, issue_source = _get_issues()

    # Pass focus_lat/focus_lng and rain override so engine scores with correct rain
    result, err = _run_prediction_thread(
        issues, focus_lat=lat, focus_lng=lng, rain_override_mm=rain_override_mm
    )

    if err:
        if cache_entry:
            stale = dict(cache_entry['result'])
            stale['cached']    = True
            stale['warning']   = f'Live fetch failed ({err}) — showing cached data'
            stale['officer_area'] = area
            return jsonify(stale)
        return jsonify({'error': err}), 500

    result['cached']       = False
    result['issue_source'] = issue_source
    result['officer_area'] = area
    result['officer_lat']  = lat
    result['officer_lng']  = lng

    # Store in a per-officer cache slot
    _cache[officer_cache_key] = {'result': result, 'ts': time.time()}

    return jsonify(result)


# ── UTILITY ROUTES ────────────────────────────────────────────
@app.route('/api/aqi')
def aqi():
    return jsonify(fetch_aqi(AQI_TOKEN))


@app.route('/api/areas')
def areas():
    return jsonify(list(DELHI_AREAS.keys()))


@app.route('/api/mock-issues')
def mock_issues_route():
    return jsonify(MOCK_ISSUES)


@app.route('/api/officer-info')
def officer_info():
    """Return the logged-in officer's profile (for frontend use)."""
    gov = session.get('gov_role')
    if not gov:
        return jsonify({'logged_in': False}), 401
    return jsonify({
        'logged_in': True,
        'name':      gov.get('name'),
        'authority': gov.get('authority'),
        'tags':      gov.get('tags', []),
        'area':      gov.get('area'),
        'lat':       gov.get('lat'),
        'lng':       gov.get('lng'),
    })


@app.route('/api/health')
def health():
    db_status = 'connected' if DATABASE_URL else 'not configured'
    if DATABASE_URL:
        try:
            issues = fetch_issues_from_postgres(DATABASE_URL, limit=1)
            db_status = f'connected ({len(issues)} test query ok)'
        except Exception as e:
            db_status = f'error: {str(e)[:60]}'
    gov = session.get('gov_role')
    return jsonify({
        'status':            'ok',
        'groq_configured':   bool(GROQ_KEY),
        'aqi_token':         AQI_TOKEN,
        'database':          db_status,
        'areas':             len(DELHI_AREAS),
        'issue_source':      'Neon Postgres' if DATABASE_URL else 'mock data',
        'officer_logged_in': bool(gov),
        'officer_area':      gov.get('area') if gov else None,
    })


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"""
╔══════════════════════════════════════════════╗
║   AreaPulse CivicAlert v4.1 — Test Server   ║
║   http://localhost:{port}                      ║
╠══════════════════════════════════════════════╣
║   DATABASE_URL: {'SET ✓  (real issues)' if DATABASE_URL else 'NOT SET (mock data)  '}     ║
║   GROQ_API_KEY: {'SET ✓' if GROQ_KEY else 'NOT SET (fallback bulletin)'}           ║
║   AQI_TOKEN:    {AQI_TOKEN[:12]}{'...' if len(AQI_TOKEN) > 12 else ''}              ║
║   CACHE_TTL:    300s                         ║
╠══════════════════════════════════════════════╣
║   Gov login: gov_rmc / gov_water /           ║
║              gov_electricity / gov_traffic   ║
║   PIN: 0000 for all                          ║
╚══════════════════════════════════════════════╝
    """)
    app.run(host='0.0.0.0', port=port, debug=True)