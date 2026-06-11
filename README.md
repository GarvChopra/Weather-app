# AreaPulse CivicAlert — Standalone Test App

All-weather civic risk prediction. 5 disaster types × 18 Delhi areas.
XGBoost ML + Open-Meteo weather + AQICN air quality + Groq AI bulletin.

## Setup (2 minutes)

```bash
cd areapulse-civicalert
pip install -r requirements.txt
python app.py
```

Open: http://localhost:5050

## What it does

Predicts civic infrastructure risk from 5 weather disasters:

| Disaster   | Civic Issues Affected              | Season        |
|------------|------------------------------------|---------------|
| Flood      | Water, sewage, pothole, electricity| June–Sept     |
| Heatwave   | Electricity, pothole, water, tree  | April–June    |
| Air Quality| Garbage, streetlight, noise        | Oct–Jan       |
| Dense Fog  | Streetlight, traffic               | Dec–Feb       |
| Cold Wave  | Water pipes, electricity           | Dec–Jan       |

## Environment Variables

```bash
# Optional — for AI bulletin (fallback works without)
export GROQ_API_KEY=your_key_here

# Optional — for real AQI (demo token gives limited data)
export AQICN_TOKEN=your_token_from_aqicn.org
```

## API Endpoints

```
GET  /                          Dashboard UI
POST /api/predict               Run full prediction
     Body: {"issues": [...], "force_refresh": false}
GET  /api/aqi                   Current Delhi AQI
GET  /api/areas                 List of Delhi areas
GET  /api/mock-issues           Sample civic issues
GET  /api/health                Config check
```

## Passing Real Issues from AreaPulse

```bash
# Export real issues from your main app
curl https://areapulse.onrender.com/api/issues?limit=500 > issues.json

# Pass to CivicAlert
curl -X POST http://localhost:5050/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"issues\": $(cat issues.json | python3 -c \"import json,sys; print(json.dumps(json.load(sys.stdin).get('issues',[]))))\")}"
```

## File Structure

```
areapulse-civicalert/
├── app.py          ← Flask server, routes, mock data
├── engine.py       ← ML engine: weather fetch, train, predict, AI bulletin
├── requirements.txt
├── README.md
├── models/         ← Auto-created on first run
│   └── civicalert_models.pkl
└── templates/
    └── index.html  ← Full dashboard UI
```

## Integration into Main AreaPulse Portal

Once tested and working:

1. Copy `engine.py` → `areapulse-portal/modules/civicalert_engine.py`
2. Add route in `areapulse-portal/app.py`:
   ```python
   from civicalert_engine import run_full_prediction as civicalert_predict

   @app.route('/gov/civicalert', methods=['POST'])
   @require_gov
   def gov_civicalert():
       issues = _get_issues_annotated()
       result = civicalert_predict(issues, groq_api_key=GROQ_KEY)
       return jsonify(result)
   ```
3. Add the dashboard template to `templates/gov/civicalert.html`
4. Add nav item to `base_gov.html`

## First Run

On first run the app trains 5 XGBoost models (~30 seconds).
After that models are cached in `models/civicalert_models.pkl`
and load in under 1 second.

Training data: synthetic data mirroring real Delhi seasonal patterns.
To retrain on real data: delete `models/civicalert_models.pkl` and
run `python -c "from engine import train_models; train_models()"`.
