# UFC Oracle v4.1

**Machine learning UFC fight prediction engine** — GitHub Actions hosted, Discord embedded alerts, event-driven scheduling.

- **Fighter Elo** (overall + striking + grappling)
- **40+ feature vector**: striking/grappling differentials, stylistic matchup, age curve, layoff, camp quality
- **Binary logistic regression** (winner) + **Multinomial LR** (method of victory)
- **Platt scaling** for calibrated probabilities
- **Market edge detection** vs Vegas closing line
- **Discord embeds**: red-themed fight card predictions, Thursday updates, Sunday recaps

**Accuracy targets**: 62–66% winner | 68–72% at 65%+ confidence | 40–45% method of victory

---

## Workflows

| Workflow | Schedule | Purpose |
|---|---|---|
| `fight-week-predictions.yml` | Monday 10AM ET | Check for Saturday event → generate picks → send Discord embed |
| `updated-predictions.yml` | Thursday 10AM ET | Re-run with updated odds/scratches → send updated embed |
| `post-event-recap.yml` | Sunday 10AM ET | Score predictions, update Elo/stats, send recap |
| `fighter-db-update.yml` | Manual | Refresh fighter stats from UFCStats.com |

---

## Setup

### 1. GitHub Secrets

Add these in **Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `DISCORD_WEBHOOK_URL` | Your Discord channel webhook URL |

### 2. Install dependencies

```bash
npm install
pip install -r python/requirements.txt
```

### 3. Build fighter database (first time)

```bash
# Scrape all active UFC fighters from UFCStats.com (~60–90 min)
npm run scrape:fighters
```

### 4. Build training dataset & train models

```bash
# Build historical dataset (2015–2025, ~4,500 fights)
npm run build-dataset

# Train winner + method models
npm run train
```

### 5. Test locally

```bash
# Generate fight card predictions for the next event
npm run alerts:picks

# Manually trigger updated predictions
npm run alerts:updated

# Manually trigger post-event recap
npm run alerts:recap
```

---

## Repository Structure

```
.github/workflows/          # GitHub Actions (Monday/Thursday/Sunday crons)
src/
  pipeline/                 # Feature engineering, prediction runner
  discord/                  # Embed builder, webhook sender
  elo/                      # Multi-dimensional Elo system
  scraper/                  # UFCStats.com scraper, fight card fetcher
  style-model/              # Stylistic classification + matchup encoding
  db/                       # SQLite database layer
  types.ts                  # Core type definitions
  index.ts                  # Main entry point
python/
  build_dataset.py          # Historical dataset builder (2015–2025)
  train_model.py            # ML training (logistic regression + calibration)
  requirements.txt
model/
  winner_model.json         # Binary LR coefficients
  method_model.json         # Multinomial LR for method of victory
  calibration.json          # Platt scaling parameters
config/
  camps.json                # Training camp quality rankings
  age_curves.json           # Per-division age curve parameters
data/
  oracle.sqlite             # Fighter database + predictions + accuracy log
```

---

## Discord Embed Design

**Fight Card (Monday/Thursday)** — Red sidebar `#D20A11`
- Main event: detailed breakdown (probabilities, method prediction, Elo comparison, style matchup)
- Co-main + main card: individual fields
- Prelims: condensed notable picks
- Running YTD record in every embed

**Post-Event Recap (Sunday)** — Gold if winning night, Red if losing
- Per-fight results with ✅/❌
- Event accuracy + YTD totals
- Method accuracy, high-conviction accuracy
- Event-by-event trend

---

## Model Architecture

```
Fighter Feature Vector (40+ features as Fighter A – Fighter B diffs)
    ↓
Logistic Regression (winner, binary, L2, C=0.5–1.0)
    ↓
Platt Scaling (calibration)
    ↓
Winner probability + Multinomial LR (method: KO/Sub/Dec)
    ↓
Edge detection vs Vegas closing line
    ↓
Discord embed
```

---

## Feature Engineering

All striking stats are **per-minute rates** (not totals) for normalization across fight durations.
Wrestling stats are **per-15-min**. A fighter averaging 6.0 sig strikes/min is comparable regardless of fight length.

Key features:
- **Elo differentials** (3 Elo ratings: overall K=32, striking K=40 on KO, grappling K=40 on Sub)
- **Stylistic matchup**: Striker vs Wrestler vs SubmissionArtist vs PressureFighter vs CounterStriker vs WellRounded
- **Age curve**: non-linear decline curves per division (Heavyweight peaks 28–35, Flyweight peaks 25–29)
- **Layoff penalty**: 0–6mo: 0%, 6–12mo: −2%, 12–18mo: −5%, 18mo+: −8%
- **Camp quality**: ~30 gyms tiered 0–3 (City Kickboxing, ATT, Sanford MMA = tier 3)

---

*UFC Oracle v4.1 — April 2026 — GitHub Actions + Discord Embeds + Event-Driven Scheduling*
