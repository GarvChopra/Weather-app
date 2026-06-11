"""
AreaPulse CivicAlert — Standalone Test App
==========================================
Run: python app.py
Open: http://localhost:5050

Tests:
  - All 5 weather disaster predictions
  - Real Open-Meteo weather data
  - Real AQICN air quality data  
  - XGBoost ML scoring
  - Groq AI bulletin (set GROQ_API_KEY env var)
  - Works with mock issues OR real issues via JSON payload
"""

import os, json, time
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

load_dotenv()  # loads .env file automatically

from engine import run_full_prediction, DELHI_AREAS, fetch_aqi, fetch_issues_from_postgres

app = Flask(__name__)

GROQ_KEY     = os.environ.get('GROQ_API_KEY', '')
AQI_TOKEN    = os.environ.get('AQICN_TOKEN', 'demo')
DATABASE_URL = os.environ.get('DATABASE_URL', '')

# ── MOCK ISSUES — realistic Delhi civic data for testing ───────
MOCK_ISSUES = [
    {'id':1, 'area':'Chandni Chowk','tag':'sewage',      'severity':'high',  'status':'open',      'description':'Open manhole on main road'},
    {'id':2, 'area':'Chandni Chowk','tag':'water',       'severity':'high',  'status':'open',      'description':'Water pipe burst near market'},
    {'id':3, 'area':'Chandni Chowk','tag':'pothole',     'severity':'medium','status':'in_progress','description':'Large pothole near metro gate'},
    {'id':4, 'area':'Kashmere Gate','tag':'sewage',      'severity':'high',  'status':'escalated', 'description':'Sewage overflow, foul smell'},
    {'id':5, 'area':'Kashmere Gate','tag':'streetlight', 'severity':'medium','status':'open',      'description':'5 streetlights out on main road'},
    {'id':6, 'area':'Paharganj',    'tag':'water',       'severity':'high',  'status':'open',      'description':'No water for 3 days'},
    {'id':7, 'area':'Paharganj',    'tag':'garbage',     'severity':'medium','status':'open',      'description':'Garbage not collected 2 weeks'},
    {'id':8, 'area':'Rohini',       'tag':'pothole',     'severity':'medium','status':'open',      'description':'Pothole near school'},
    {'id':9, 'area':'Dwarka',       'tag':'electricity', 'severity':'high',  'status':'open',      'description':'Daily power cuts 3-5pm'},
    {'id':10,'area':'Dwarka',       'tag':'water',       'severity':'medium','status':'open',      'description':'Low water pressure'},
    {'id':11,'area':'Saket',        'tag':'tree',        'severity':'low',   'status':'open',      'description':'Tree branches touching power lines'},
    {'id':12,'area':'Mayur Vihar',  'tag':'sewage',      'severity':'high',  'status':'open',      'description':'Drain blocked, standing water'},
    {'id':13,'area':'Karol Bagh',   'tag':'traffic',     'severity':'medium','status':'open',      'description':'Traffic signal not working'},
    {'id':14,'area':'Mukherjee Nagar','tag':'noise',     'severity':'low',   'status':'open',      'description':'Construction noise after 10pm'},
    {'id':15,'area':'Okhla',        'tag':'garbage',     'severity':'high',  'status':'escalated', 'description':'Industrial waste dumped illegally'},
]

# Cache prediction results for 5 minutes
_cache = {'result': None, 'ts': 0}
CACHE_TTL = 120

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Run full prediction.
    POST body (optional):
    {
        "issues": [...],   // pass real issues or leave empty for mock
        "force_refresh": false
    }
    """
    global _cache

    body = request.get_json(silent=True) or {}
    force = body.get('force_refresh', False)

    # Use cached result if fresh
    if not force and _cache['result'] and (time.time() - _cache['ts']) < CACHE_TTL:
        result = dict(_cache['result'])
        result['cached'] = True
        result['cache_age_s'] = int(time.time() - _cache['ts'])
        return jsonify(result)

    # Priority: 1) issues from POST body, 2) Postgres DB, 3) mock issues
    issues = body.get('issues', None)

    if issues is None:
        if DATABASE_URL:
            try:
                issues = fetch_issues_from_postgres(DATABASE_URL)
                issue_source = f'Neon Postgres ({len(issues)} real issues)'
            except Exception as db_err:
                print(f"[db] {db_err} — falling back to mock issues")
                issues = MOCK_ISSUES
                issue_source = 'mock (DB connection failed)'
        else:
            issues = MOCK_ISSUES
            issue_source = 'mock (DATABASE_URL not set)'
    else:
        issue_source = f'POST body ({len(issues)} issues)'

    # Run prediction in thread with HARD 12s timeout
    import threading
    result_box = [None]
    error_box  = [None]

    def _run():
        try:
            result_box[0] = run_full_prediction(
                open_issues=issues,
                groq_api_key=GROQ_KEY,
                aqi_token=AQI_TOKEN,
            )
        except Exception as e:
            error_box[0] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=12)  # Hard 12-second deadline — never hang past this

    if t.is_alive():
        # Thread still running after 12s = API is hanging
        err_msg = "Weather API timed out after 12s. Check internet connection."
        print(f"[predict] TIMEOUT: {err_msg}")
    elif error_box[0]:
        err_msg = error_box[0]
        print(f"[predict] ERROR: {err_msg}")
    else:
        result = result_box[0]
        result['cached']       = False
        result['issue_source'] = issue_source
        _cache['result'] = result
        _cache['ts']     = time.time()
        return jsonify(result)

    # Execution reaches here only on timeout or error
    # Return stale cache if available, otherwise error
    if _cache['result']:
        stale = dict(_cache['result'])
        stale['cached']      = True
        stale['cache_age_s'] = int(time.time() - _cache['ts'])
        stale['warning']     = f'Live fetch failed ({err_msg}) — showing last known data'
        return jsonify(stale)
    return jsonify({'error': err_msg}), 500

@app.route('/api/aqi')
def aqi():
    return jsonify(fetch_aqi(AQI_TOKEN))

@app.route('/api/areas')
def areas():
    return jsonify(list(DELHI_AREAS.keys()))

@app.route('/api/mock-issues')
def mock_issues():
    return jsonify(MOCK_ISSUES)

@app.route('/api/health')
def health():
    db_status = 'connected' if DATABASE_URL else 'not configured'
    if DATABASE_URL:
        try:
            issues = fetch_issues_from_postgres(DATABASE_URL, limit=1)
            db_status = f'connected ({len(issues)} test query ok)'
        except Exception as e:
            db_status = f'error: {str(e)[:60]}'
    return jsonify({
        'status':           'ok',
        'groq_configured':  bool(GROQ_KEY),
        'aqi_token':        AQI_TOKEN,
        'database':         db_status,
        'areas':            len(DELHI_AREAS),
        'issue_source':     'Neon Postgres' if DATABASE_URL else 'mock data',
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"""
╔══════════════════════════════════════════╗
║   AreaPulse CivicAlert — Test Server     ║
║   http://localhost:{port}                  ║
╠══════════════════════════════════════════╣
║   DATABASE_URL: {'SET ✓  (real issues)' if DATABASE_URL else 'NOT SET (using mock)  '}   ║
║   GROQ_API_KEY: {'SET ✓' if GROQ_KEY else 'NOT SET (fallback)  '}              ║
║   AQI_TOKEN:    {AQI_TOKEN[:12]}{'...' if len(AQI_TOKEN)>12 else ''}            ║
╚══════════════════════════════════════════╝
    """)
    app.run(host='0.0.0.0', port=port, debug=True)