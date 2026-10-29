"""
Admission Predictor – Flask app serving adaboost.pkl
Deploy on Render:  Build:  pip install -r requirements.txt
                   Start:  gunicorn app:app
"""
import os
import warnings

import joblib
import pandas as pd
from flask import Flask, render_template_string, request

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "adaboost.pkl")
model = joblib.load(MODEL_PATH)

# The model was trained on 19 columns, but only the 11 below affect its output
# (the other 8 have zero feature importance), so they are filled with 0.
MODEL_COLUMNS = list(model.feature_names_in_)

# key, label, min, max, step, default, hint
FIELDS = {
    "Academics": [
        ("high_school_gpa", "High school GPA", 0, 4, 0.01, 3.5, "Out of 4.0"),
        ("sat_score", "SAT score", 400, 1600, 1, 1200, "Out of 1600"),
        ("ap_courses", "AP courses taken", 0, 15, 1, 3, "Count"),
    ],
    "Activities": [
        ("extracurricular_count", "Extracurriculars", 0, 15, 1, 3, "Clubs, sports, arts"),
        ("volunteer_hours", "Volunteer hours", 0, 500, 1, 80, "Total hours"),
        ("leadership_positions", "Leadership roles", 0, 10, 1, 1, "Captain, president, etc."),
        ("coding_projects", "Coding projects", 0, 20, 1, 2, "Completed projects"),
        ("online_certifications", "Online certifications", 0, 20, 1, 2, "Completed courses"),
    ],
    "Evaluations": [
        ("essay_score", "Essay score", 0, 100, 1, 70, "Out of 100"),
        ("recommendation_score", "Recommendation score", 0, 100, 1, 70, "Out of 100"),
        ("interview_score", "Interview score", 0, 100, 1, 70, "Out of 100"),
    ],
}
ALL_FIELDS = [f for group in FIELDS.values() for f in group]

# ----------------------------------------------------------------------------
# STYLE
# ----------------------------------------------------------------------------
STYLE = """
:root{
  --ink:#16193F;        /* deep indigo */
  --ink-2:#262B63;
  --dusk:#5B5FC7;       /* periwinkle */
  --mist:#E9E8FB;       /* lavender mist */
  --paper:#FBFAFF;
  --brass:#F2B134;      /* brass gold */
  --brass-d:#C98A0B;
  --rose:#E4577A;
  --mint:#2FB391;
  --text:#1D2050;
  --muted:#6A6D9A;
  --radius:18px;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  font-family:'DM Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  color:var(--text);
  background:
    radial-gradient(900px 500px at 8% -10%, rgba(242,177,52,.22), transparent 60%),
    radial-gradient(900px 600px at 100% 0%, rgba(91,95,199,.28), transparent 60%),
    linear-gradient(180deg,#EFEEFF 0%, #F8F7FF 55%, #FFFFFF 100%);
  background-attachment:fixed;
  min-height:100vh;
  line-height:1.5;
}
h1,h2,h3{font-family:'Bricolage Grotesque','DM Sans',system-ui,sans-serif;letter-spacing:-.02em}

.shell{
  max-width:1180px;margin:0 auto;padding:32px 20px 56px;
  display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);gap:28px;align-items:start;
}

/* ---------- hero panel ---------- */
.hero{
  position:sticky;top:24px;
  background:linear-gradient(165deg,var(--ink) 0%,var(--ink-2) 100%);
  color:#fff;border-radius:28px;overflow:hidden;
  box-shadow:0 30px 60px -25px rgba(22,25,63,.65);
}
.hero-art{display:block;width:100%;height:auto}
.hero-body{padding:8px 32px 34px}
.hero h1{font-size:2.35rem;line-height:1.08;font-weight:800;margin-bottom:12px}
.hero p{color:#C9CBF2;font-size:1rem;max-width:36ch}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}
.chip{
  font-size:.82rem;padding:6px 12px;border-radius:999px;
  background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);color:#E6E7FF;
}

/* ---------- form card ---------- */
.card{
  background:rgba(255,255,255,.88);
  backdrop-filter:blur(10px);
  border:1px solid #E1E0F7;border-radius:28px;
  padding:30px 30px 34px;
  box-shadow:0 24px 50px -30px rgba(38,43,99,.45);
}
fieldset{border:0;margin-bottom:26px}
legend{
  font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.2rem;
  color:var(--ink);padding-bottom:10px;margin-bottom:6px;
  border-bottom:2px solid var(--mist);width:100%;
}
.row{display:grid;grid-template-columns:1fr 1fr;gap:6px 22px}
.field{padding:12px 0 6px}
.field label{font-weight:600;font-size:.95rem}
.hint{font-size:.78rem;color:var(--muted);margin-top:6px}

input[type=number]{
  width:100%;margin-top:8px;padding:12px 14px;
  font-family:'Bricolage Grotesque','DM Sans',sans-serif;font-weight:700;font-size:1.1rem;color:var(--ink);
  background:#fff;border:2px solid #DAD9F3;border-radius:12px;outline:none;
  transition:border-color .12s ease, box-shadow .12s ease;
  -moz-appearance:textfield;appearance:textfield;
}
input[type=number]::-webkit-outer-spin-button,
input[type=number]::-webkit-inner-spin-button{opacity:1}
input[type=number]:hover{border-color:#B9B8E8}
input[type=number]:focus{border-color:var(--dusk);box-shadow:0 0 0 4px rgba(91,95,199,.18)}
input[type=number]:invalid{border-color:var(--rose)}

.btn{
  width:100%;border:0;cursor:pointer;
  font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.1rem;
  padding:16px 22px;border-radius:14px;color:var(--ink);
  background:linear-gradient(180deg,#FFC956,var(--brass));
  box-shadow:0 10px 24px -10px rgba(201,138,11,.8), inset 0 -3px 0 var(--brass-d);
  transition:transform .12s ease, box-shadow .12s ease;
}
.btn:hover{transform:translateY(-1px)}
.btn:active{transform:translateY(1px)}
.btn:focus-visible{outline:3px solid var(--dusk);outline-offset:3px}

/* ---------- result ---------- */
.result{
  display:grid;grid-template-columns:auto 1fr;gap:24px;align-items:center;
  margin-bottom:28px;padding:22px 24px;border-radius:22px;
  background:var(--paper);border:2px solid var(--tone);
  animation:pop .45s cubic-bezier(.2,.9,.3,1.2);
}
.result.good{--tone:var(--mint)}
.result.bad{--tone:var(--rose)}
.ring{
  --pct:0;width:132px;height:132px;border-radius:50%;display:grid;place-items:center;
  background:conic-gradient(var(--tone) calc(var(--pct)*1%), #E4E3F6 0);
}
.ring span{
  width:100px;height:100px;border-radius:50%;background:#fff;display:grid;place-items:center;
  font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:1.7rem;color:var(--ink);
}
.result h2{font-size:1.5rem;color:var(--ink);margin-bottom:4px}
.result p{color:var(--muted);font-size:.95rem}
.error{background:#FFF0F3;border:1px solid var(--rose);color:#8B1E3F;padding:12px 16px;border-radius:12px;margin-bottom:20px}
@keyframes pop{from{opacity:0;transform:scale(.96)}to{opacity:1;transform:none}}

.foot{grid-column:1/-1;text-align:center;color:var(--muted);font-size:.82rem;margin-top:6px}

/* ---------- responsive ---------- */
@media (max-width:900px){
  .shell{grid-template-columns:1fr}
  .hero{position:relative;top:0}
  .hero h1{font-size:1.9rem}
}
@media (max-width:560px){
  .row{grid-template-columns:1fr}
  .card{padding:22px 18px 26px}
  .result{grid-template-columns:1fr;justify-items:center;text-align:center}
}
@media (prefers-reduced-motion:reduce){
  *{animation:none!important;transition:none!important}
}
"""

# Inline SVG illustration (no external image dependency, so it never breaks on deploy)
HERO_SVG = """
<svg class="hero-art" viewBox="0 0 600 380" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Illustration of a university hall at dusk with a graduation cap">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#262B63"/><stop offset=".65" stop-color="#6B63C9"/>
      <stop offset="1" stop-color="#F2B134"/>
    </linearGradient>
    <linearGradient id="hall" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#F6F1FF"/><stop offset="1" stop-color="#CFCBF0"/>
    </linearGradient>
  </defs>
  <rect width="600" height="380" fill="url(#sky)"/>
  <g fill="#fff" opacity=".85">
    <circle cx="70" cy="50" r="2"/><circle cx="150" cy="28" r="1.6"/><circle cx="260" cy="60" r="1.8"/>
    <circle cx="470" cy="38" r="2"/><circle cx="540" cy="86" r="1.5"/><circle cx="395" cy="70" r="1.4"/>
  </g>
  <circle cx="500" cy="120" r="34" fill="#FFE9A8" opacity=".95"/>
  <path d="M0 300 Q150 250 300 285 T600 270 V380 H0Z" fill="#1B1F52"/>
  <!-- hall -->
  <g>
    <rect x="130" y="205" width="340" height="110" fill="url(#hall)"/>
    <polygon points="115,205 300,140 485,205" fill="#EDE8FF"/>
    <polygon points="115,205 300,140 485,205 470,205 300,150 130,205" fill="#C9C3EC"/>
    <rect x="270" y="95" width="60" height="55" fill="#F6F1FF"/>
    <polygon points="262,95 300,62 338,95" fill="#16193F"/>
    <circle cx="300" cy="120" r="14" fill="#16193F"/>
    <circle cx="300" cy="120" r="11" fill="#FFF7DC"/>
    <path d="M300 120V112M300 120L306 123" stroke="#16193F" stroke-width="2" stroke-linecap="round"/>
    <g fill="#E3DEF7">
      <rect x="150" y="212" width="14" height="103"/><rect x="190" y="212" width="14" height="103"/>
      <rect x="230" y="212" width="14" height="103"/><rect x="356" y="212" width="14" height="103"/>
      <rect x="396" y="212" width="14" height="103"/><rect x="436" y="212" width="14" height="103"/>
    </g>
    <path d="M270 315V262a30 30 0 0 1 60 0V315Z" fill="#16193F"/>
    <g fill="#F2B134" opacity=".9">
      <rect x="250" y="235" width="0" height="0"/>
    </g>
  </g>
  <rect x="0" y="312" width="600" height="68" fill="#12153A"/>
  <path d="M300 315 L250 380 H350Z" fill="#F2B134" opacity=".35"/>
  <!-- graduation cap -->
  <g transform="translate(78 120) rotate(-12)">
    <polygon points="0,30 60,5 120,30 60,55" fill="#F2B134"/>
    <path d="M25 42v22c0 8 70 8 70 0V42L60 55Z" fill="#C98A0B"/>
    <path d="M120 30v34" stroke="#FFE9A8" stroke-width="3" stroke-linecap="round"/>
    <circle cx="120" cy="68" r="5" fill="#FFE9A8"/>
  </g>
</svg>
"""

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Admission Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700;800&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <style>{{ style|safe }}</style>
</head>
<body>
  <main class="shell">
    <aside class="hero">
      {{ art|safe }}
      <div class="hero-body">
        <h1>Check your admission chances</h1>
        <p>Enter your grades, activities and interview results. An AdaBoost model trained on past applicants estimates how likely an admission is.</p>
        <div class="chips">
          <span class="chip">AdaBoost classifier</span>
          <span class="chip">11 profile signals</span>
          <span class="chip">Instant result</span>
        </div>
      </div>
    </aside>

    <section class="card">
      {% if error %}<div class="error" role="alert">{{ error }}</div>{% endif %}

      {% if result %}
      <div class="result {{ 'good' if result.admitted else 'bad' }}" role="status">
        <div class="ring" style="--pct: {{ result.pct }}"><span>{{ result.pct }}%</span></div>
        <div>
          <h2>{{ 'Likely to be admitted' if result.admitted else 'Less likely to be admitted' }}</h2>
          <p>The model gives this profile a {{ result.pct }}% chance of admission. Change any value and predict again to compare.</p>
        </div>
      </div>
      {% endif %}

      <form method="post" action="/">
        {% for group, fields in groups.items() %}
        <fieldset>
          <legend>{{ group }}</legend>
          <div class="row">
            {% for key, label, lo, hi, step, default, hint in fields %}
            <div class="field">
              <label for="{{ key }}">{{ label }}</label>
              <input type="number" id="{{ key }}" name="{{ key }}" inputmode="decimal"
                     min="{{ lo }}" max="{{ hi }}" step="{{ step }}" value="{{ values[key] }}" required>
              <div class="hint">{{ hint }} &middot; range {{ lo }} to {{ hi }}</div>
            </div>
            {% endfor %}
          </div>
        </fieldset>
        {% endfor %}
        <button class="btn" type="submit">Predict admission</button>
      </form>
    </section>

    <p class="foot">Predictions are statistical estimates and not a guarantee of any real admission decision.</p>
  </main>

</body>
</html>
"""


def default_values():
    return {key: default for key, _, _, _, _, default, _ in ALL_FIELDS}


def parse_form(form):
    """Validate and clamp submitted values. Returns (values, error)."""
    values = default_values()
    for key, label, lo, hi, step, default, _ in ALL_FIELDS:
        raw = form.get(key, "")
        try:
            num = float(raw)
        except (TypeError, ValueError):
            return values, f"'{label}' needs a number."
        if not lo <= num <= hi:
            return values, f"'{label}' must be between {lo} and {hi}."
        values[key] = int(num) if float(step).is_integer() else round(num, 2)
    return values, None


def predict(values):
    row = {col: 0 for col in MODEL_COLUMNS}
    row.update(values)
    frame = pd.DataFrame([row])[MODEL_COLUMNS]
    prob = float(model.predict_proba(frame)[0][1])
    return {"admitted": prob >= 0.5, "pct": round(prob * 100)}


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    values, result, error = default_values(), None, None
    if request.method == "POST":
        values, error = parse_form(request.form)
        if error is None:
            result = predict(values)
    return render_template_string(
        PAGE, style=STYLE, art=HERO_SVG, groups=FIELDS,
        values=values, result=result, error=error,
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)