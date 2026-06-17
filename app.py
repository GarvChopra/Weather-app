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
from io import BytesIO
from datetime import datetime

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
# Per-officer cache: keys are "lat,lng:rain_override" strings
# 'default' key used by /api/predict (non-location route)
_cache = {}
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
@app.route('/api/chat', methods=['POST'])
def chat():
    """
    AI chatbot for CivicAlert detail page.
    Receives area context + question, returns Groq answer.
    No CORS issues — runs server-side.
    """
    import requests as _req
 
    body     = request.get_json(silent=True) or {}
    area     = body.get('area', 'Delhi')
    label    = body.get('label', '')
    score    = body.get('score', 0)
    cond     = body.get('weather_condition', '')
    temp     = body.get('weather_temp', 0)
    rain     = body.get('weather_rain', 0)
    question = body.get('question', '').strip()
    history  = body.get('history', [])
 
    if not question:
        return jsonify({'error': 'No question provided'}), 400
 
    if not GROQ_KEY:
        return jsonify({'reply': 'AI not configured. Add GROQ_API_KEY to environment variables.'}), 200
 
    LABEL_NAMES = {
        'label_flood':           'Waterlogging / Flood',
        'label_pothole_worsen':  'Pothole & Road Damage',
        'label_sewage_overflow': 'Sewage Overflow',
        'label_garbage_flood':   'Garbage Flooding',
        'label_elec_hazard':     'Electrical Hazard',
    }
 
    system_prompt = (
        f"You are AreaPulse CivicAlert, an AI assistant for Delhi municipal officers. "
        f"Current context: Area = {area}, Threat = {LABEL_NAMES.get(label, label)}, "
        f"Risk Score = {score}/100. "
        f"Weather: {cond}, {temp}°C, rain = {rain}mm/hr. "
        f"Give SHORT, direct, actionable answers in 2-3 sentences maximum. "
        f"No markdown, no bullet points, no headers. Plain text only. "
        f"Focus on what the officer should do RIGHT NOW."
    )
 
    # Build messages — include last 6 turns for context
    messages = []
    for h in history[-6:]:
        role = h.get('role', 'user')
        if role in ('user', 'assistant'):
            messages.append({'role': role, 'content': str(h.get('content', ''))})
    messages.append({'role': 'user', 'content': question})
 
    try:
        resp = _req.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {GROQ_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'model':       'meta-llama/llama-4-scout-17b-16e-instruct',
                'messages':    [{'role': 'system', 'content': system_prompt}] + messages,
                'max_tokens':  120,
                'temperature': 0.4,
            },
            timeout=12,
        )
        resp.raise_for_status()
        reply = resp.json()['choices'][0]['message']['content'].strip()
        return jsonify({'reply': reply})
    except Exception as e:
        print(f"[chat] Groq failed: {e}")
        return jsonify({'reply': f'AI temporarily unavailable. For {area}: deploy {LABEL_NAMES.get(label,"relevant")} response team immediately if risk score is above 50.'}), 200


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
    _default_cache = _cache.get('default')
    if not force and _default_cache and (time.time() - _default_cache['ts']) < CACHE_TTL:
        result = dict(_default_cache['result'])
        result['cached']      = True
        result['cache_age_s'] = int(time.time() - _default_cache['ts'])
        return jsonify(result)

    issues, issue_source = _get_issues()

    result, err = _run_prediction_thread(issues)

    if err:
        if _default_cache:
            stale = dict(_default_cache['result'])
            stale['cached']      = True
            stale['cache_age_s'] = int(time.time() - _default_cache['ts'])
            stale['warning']     = f'Live fetch failed ({err}) — showing last known data'
            return jsonify(stale)
        return jsonify({'error': err}), 500

    result['cached']       = False
    result['issue_source'] = issue_source
    _cache['default'] = {'result': result, 'ts': time.time()}
    return jsonify(result)


@app.route('/api/export-pdf', methods=['POST'])
def export_pdf():
    """
    Generate AI-written daily PDF report for gov officials.
    Uses ReportLab for PDF generation + Groq for AI summary.
    Falls back to plain HTML-based PDF if ReportLab not installed.
    """
    from io import BytesIO
    from datetime import datetime
 
    body             = request.get_json(silent=True) or {}
    areas            = body.get('areas', [])
    weather          = body.get('weather', {})
    aqi_data         = body.get('aqi', {})
    ai_bulletin      = body.get('ai_bulletin', '')
    issues_analysed  = body.get('issues_analysed', 0)
    generated_at     = body.get('generated_at', datetime.now().isoformat())
    today            = datetime.now().strftime('%d %B %Y')
    time_str         = datetime.now().strftime('%H:%M IST')
 
    # ── AI-generated executive summary ──────────────────────
    exec_summary = ai_bulletin  # default
    if GROQ_KEY and areas:
        try:
            import requests as _req
            top5 = sorted(areas, key=lambda x: x.get('overall_risk', 0), reverse=True)[:5]
            area_lines = '\n'.join(
                f"- {a['area']}: risk={a.get('overall_risk',0)}/100, "
                f"drain={a.get('drain_quality','—')}, issues={a.get('open_issues',0)}"
                for a in top5
            )
            prompt = (
                f"You are AreaPulse CivicAlert. Write a professional 3-sentence executive summary "
                f"for a Delhi municipal PDF report dated {today}.\n\n"
                f"Weather: {weather.get('current_condition','—')}, {weather.get('curr_temp','—')}°C, "
                f"rain={weather.get('curr_rain',0)}mm/hr. AQI={aqi_data.get('aqi','—') if aqi_data else '—'}.\n"
                f"Total issues analysed: {issues_analysed}.\n"
                f"Top risk areas:\n{area_lines}\n\n"
                f"Write for a municipal commissioner. 3 sentences only. No markdown. No bullet points."
            )
            resp = _req.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={'Authorization': f'Bearer {GROQ_KEY}', 'Content-Type': 'application/json'},
                json={
                    'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 150,
                    'temperature': 0.3,
                },
                timeout=10,
            )
            resp.raise_for_status()
            exec_summary = resp.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"[pdf] Groq summary failed: {e}")
 
    # ── Issue category breakdown ────────────────────────────
    LABEL_NAMES = {
        'label_flood':           'Waterlogging / Flood',
        'label_pothole_worsen':  'Pothole & Road Damage',
        'label_sewage_overflow': 'Sewage Overflow',
        'label_garbage_flood':   'Garbage Flooding',
        'label_elec_hazard':     'Electrical Hazard',
    }
 
    # ── Try ReportLab first (best quality) ──────────────────
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
 
        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=20*mm, rightMargin=20*mm,
            topMargin=18*mm, bottomMargin=18*mm,
        )
 
        BLUE   = colors.HexColor('#0066FF')
        NAVY   = colors.HexColor('#1B1B1B')
        GREY   = colors.HexColor('#6B7280')
        LIGHT  = colors.HexColor('#EBF4FF')
        RED    = colors.HexColor('#c62828')
        WHITE  = colors.white
 
        styles = getSampleStyleSheet()
        H1  = ParagraphStyle('H1',  fontSize=18, fontName='Helvetica-Bold', textColor=BLUE,   spaceAfter=4)
        H2  = ParagraphStyle('H2',  fontSize=12, fontName='Helvetica-Bold', textColor=NAVY,   spaceBefore=14, spaceAfter=6)
        SUB = ParagraphStyle('SUB', fontSize=9,  fontName='Helvetica',      textColor=GREY,   spaceAfter=10)
        BOD = ParagraphStyle('BOD', fontSize=10, fontName='Helvetica',      textColor=NAVY,   spaceAfter=8, leading=15)
        SML = ParagraphStyle('SML', fontSize=8,  fontName='Helvetica',      textColor=GREY)
 
        story = []
 
        # Header
        story.append(Paragraph('AreaPulse CivicAlert', H1))
        story.append(Paragraph(f'Daily Municipal Intelligence Report — {today} · {time_str}', SUB))
        story.append(HRFlowable(width='100%', thickness=1, color=BLUE, spaceAfter=12))
 
        # Weather summary row
        wx_data = [
            ['Condition', 'Temperature', 'Rain Now', 'Forecast 6h', 'AQI'],
            [
                weather.get('current_condition', '—'),
                f"{weather.get('curr_temp', '—')}°C",
                f"{weather.get('curr_rain', 0)}mm/hr",
                f"{weather.get('rain_next_6h', 0)}mm",
                str(aqi_data.get('aqi', '—')) if aqi_data else '—',
            ]
        ]
        wx_table = Table(wx_data, colWidths=[38*mm]*5)
        wx_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), BLUE),
            ('TEXTCOLOR',  (0,0), (-1,0), WHITE),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 9),
            ('BACKGROUND', (0,1), (-1,-1), LIGHT),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT, WHITE]),
            ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#BFDBFE')),
            ('TOPPADDING',    (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(wx_table)
        story.append(Spacer(1, 10))
 
        # AI Executive Summary
        story.append(Paragraph('Executive Summary', H2))
        story.append(Paragraph(exec_summary, BOD))
        story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#BFDBFE'), spaceAfter=8))
 
        # Stats row
        story.append(Paragraph('Key Statistics', H2))
        at_risk = sum(1 for a in areas if a.get('overall_risk', 0) >= 1)
        critical = sum(1 for a in areas if a.get('overall_risk', 0) >= 50)
        stats_data = [
            ['Areas Monitored', 'Issues Analysed', 'Areas At Risk', 'Critical (≥50)'],
            [str(len(areas)), str(issues_analysed), str(at_risk), str(critical)],
        ]
        st = Table(stats_data, colWidths=[42*mm]*4)
        st.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), NAVY),
            ('TEXTCOLOR',  (0,0), (-1,0), WHITE),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME',   (0,1), (-1,1), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 9),
            ('FONTSIZE',   (0,1), (-1,-1), 16),
            ('TEXTCOLOR',  (0,1), (-1,1), BLUE),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,1), (-1,-1), LIGHT),
            ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#BFDBFE')),
            ('TOPPADDING',    (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(st)
        story.append(Spacer(1, 14))
 
        # Top risk areas table
        story.append(Paragraph('Area Risk Assessment', H2))
        sorted_areas = sorted(areas, key=lambda x: x.get('overall_risk', 0), reverse=True)
 
        area_rows = [['Area', 'Zone', 'Risk Score', 'Top Threat', 'Drainage', 'Open Issues', 'Status']]
        for a in sorted_areas[:30]:  # top 30
            score = a.get('overall_risk', 0)
            top   = max(a.get('scores', {}).items(), key=lambda x: x[1], default=(None,0))
            threat = LABEL_NAMES.get(top[0], '—') if top[0] else '—'
            status = 'CRITICAL' if score >= 75 else 'HIGH' if score >= 50 else 'AT RISK' if score >= 25 else 'LOW'
            area_rows.append([
                a.get('area', '—'),
                a.get('zone', '—'),
                str(score),
                threat[:22],
                a.get('drain_quality', '—'),
                str(a.get('open_issues', 0)),
                status,
            ])
 
        col_w = [38*mm, 16*mm, 18*mm, 42*mm, 20*mm, 18*mm, 18*mm]
        at = Table(area_rows, colWidths=col_w, repeatRows=1)
        row_styles = [
            ('BACKGROUND', (0,0), (-1,0), NAVY),
            ('TEXTCOLOR',  (0,0), (-1,0), WHITE),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 8),
            ('ALIGN',      (2,0), (2,-1), 'CENTER'),
            ('ALIGN',      (5,0), (5,-1), 'CENTER'),
            ('ALIGN',      (6,0), (6,-1), 'CENTER'),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('GRID',       (0,0), (-1,-1), 0.3, colors.HexColor('#BFDBFE')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, LIGHT]),
            ('TOPPADDING',    (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]
        # Highlight critical rows red
        for i, a in enumerate(sorted_areas[:30], 1):
            if a.get('overall_risk', 0) >= 75:
                row_styles.append(('TEXTCOLOR', (2,i), (2,i), RED))
                row_styles.append(('FONTNAME',  (2,i), (2,i), 'Helvetica-Bold'))
        at.setStyle(TableStyle(row_styles))
        story.append(at)
        story.append(Spacer(1, 14))
 
        # Issue category breakdown
        story.append(Paragraph('Issue Category Breakdown', H2))
        cat_counts = {}
        for a in areas:
            for iss in a.get('affected_issues', []):
                tag = iss.get('tag', 'other')
                sev = iss.get('severity', 'low')
                cat_counts.setdefault(tag, {'high':0,'medium':0,'low':0,'total':0})
                cat_counts[tag][sev] = cat_counts[tag].get(sev, 0) + 1
                cat_counts[tag]['total'] += 1
 
        if cat_counts:
            cat_rows = [['Category', 'High', 'Medium', 'Low', 'Total']]
            for tag, counts in sorted(cat_counts.items(), key=lambda x: -x[1]['total']):
                cat_rows.append([
                    tag.title(), str(counts.get('high',0)),
                    str(counts.get('medium',0)), str(counts.get('low',0)),
                    str(counts.get('total',0))
                ])
            ct = Table(cat_rows, colWidths=[60*mm,25*mm,25*mm,25*mm,25*mm])
            ct.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), BLUE),
                ('TEXTCOLOR',  (0,0), (-1,0), WHITE),
                ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE',   (0,0), (-1,-1), 9),
                ('ALIGN',      (1,0), (-1,-1), 'CENTER'),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, LIGHT]),
                ('GRID',       (0,0), (-1,-1), 0.3, colors.HexColor('#BFDBFE')),
                ('TOPPADDING',    (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ]))
            story.append(ct)
        else:
            story.append(Paragraph('No issue breakdown available.', SML))
 
        # Footer
        story.append(Spacer(1, 16))
        story.append(HRFlowable(width='100%', thickness=0.5, color=GREY))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f'Generated by AreaPulse CivicAlert · AI-powered by XGBoost + Groq Llama-4-Scout · '
            f'{today} {time_str} · Confidential — For Authorised Municipal Officers Only',
            SML
        ))
 
        doc.build(story)
        buf.seek(0)
        return buf.read(), 200, {
            'Content-Type': 'application/pdf',
            'Content-Disposition': f'attachment; filename=CivicAlert_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
        }
 
    except ImportError:
        # ── Fallback: HTML response the browser can print as PDF ──
        print("[pdf] ReportLab not installed — returning HTML fallback")
        rows_html = ''
        for a in sorted(areas, key=lambda x: x.get('overall_risk',0), reverse=True)[:30]:
            score = a.get('overall_risk', 0)
            top   = max(a.get('scores', {}).items(), key=lambda x: x[1], default=(None,0))
            threat = LABEL_NAMES.get(top[0], '—') if top[0] else '—'
            color = '#c62828' if score >= 75 else '#e65100' if score >= 50 else '#1B1B1B'
            rows_html += f"""<tr>
                <td>{a.get('area','—')}</td>
                <td style='text-align:center'>{a.get('zone','—')}</td>
                <td style='text-align:center;color:{color};font-weight:700'>{score}</td>
                <td>{threat}</td>
                <td style='text-align:center'>{a.get('drain_quality','—')}</td>
                <td style='text-align:center'>{a.get('open_issues',0)}</td>
            </tr>"""
 
        html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
        <title>CivicAlert Report {today}</title>
        <style>
          body{{font-family:Arial,sans-serif;font-size:11px;color:#111;padding:20px}}
          h1{{color:#0066FF;font-size:20px;margin-bottom:4px}}
          h2{{font-size:13px;color:#111;margin:16px 0 8px}}
          .sub{{color:#6B7280;font-size:10px;margin-bottom:16px}}
          table{{width:100%;border-collapse:collapse;margin-bottom:14px}}
          th{{background:#0066FF;color:#fff;padding:6px 8px;font-size:10px;text-align:left}}
          td{{padding:5px 8px;border-bottom:1px solid #e5e7eb;font-size:10px}}
          tr:nth-child(even){{background:#EBF4FF}}
          .summary{{background:#EBF4FF;border-left:3px solid #0066FF;padding:10px 14px;margin-bottom:14px;font-size:11px;line-height:1.6}}
          .footer{{color:#9ca3af;font-size:9px;margin-top:20px;padding-top:8px;border-top:1px solid #e5e7eb}}
          @media print{{body{{padding:0}}}}
        </style>
        <script>window.onload=function(){{window.print()}}</script>
        </head><body>
        <h1>AreaPulse CivicAlert</h1>
        <div class="sub">Daily Municipal Intelligence Report — {today} · {time_str}</div>
        <h2>Executive Summary</h2>
        <div class="summary">{exec_summary}</div>
        <h2>Weather · {weather.get('current_condition','—')} · {weather.get('curr_temp','—')}°C · Rain {weather.get('curr_rain',0)}mm/hr · AQI {aqi_data.get('aqi','—') if aqi_data else '—'}</h2>
        <h2>Area Risk Assessment (Top 30)</h2>
        <table><thead><tr>
          <th>Area</th><th>Zone</th><th>Risk Score</th><th>Top Threat</th><th>Drainage</th><th>Open Issues</th>
        </tr></thead><tbody>{rows_html}</tbody></table>
        <div class="footer">Generated by AreaPulse CivicAlert · AI: XGBoost + Groq Llama-4-Scout · {today} {time_str} · Confidential</div>
        </body></html>"""
 
        return html, 200, {
            'Content-Type': 'text/html',
            'Content-Disposition': f'inline; filename=CivicAlert_{datetime.now().strftime("%Y%m%d")}.html'
        }

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