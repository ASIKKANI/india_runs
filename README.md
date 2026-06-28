# Redrob AI Candidate Ranking Engine — Track 1 (Data & AI Challenge)

This repository contains the production implementation of the Candidate Ranking Engine designed to find and rank the top 100 candidate profiles from a 100,000 candidate pool (`candidates.jsonl`) for a **"Senior AI Engineer — Founding Team"** role at Redrob AI.

The engine completes the entire scoring, filtering, ranking, and reasoning compilation in **under 14 seconds** on a standard CPU with **0% honeypot rate** in the shortlist (100% compliant with submission spec rules).
 
---

## 🏆 System Architecture: Multi-Stage Cascade Ranking

The architecture is built for ultra-fast, local, and 100% offline retrieval. It runs in four distinct stages:

```
[100,000 Candidates Pool]
         │
         ▼
┌──────────────────────────────────────────────┐
│ Stage 1: Honeypot & Anomaly Filtering        │ --> Immediately filters fraud & inconsistencies (Score = 0.0)
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ Stage 2: Dense Semantic Vector Match         │ --> Local Sentence-Transformer all-MiniLM-L6-v2 dot-product match
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ Stage 3: Target Role Heuristic Multipliers   │ --> YoE fit, location, notice period, job-hopper & services penalties
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ Stage 4: Sorting & Dynamic Reasoning         │ --> Sorts: score DESC, candidate_id ASC. Compiles 1-2 sentence rationales.
└──────────────────────────────────────────────┘
         │
         ▼
[Top 100 Shortlist (submission.csv)]
```

### Key Highlights
1. **Zero-Honeypot Guarantee:** Implements strict validation checks in `utils/honeypots.py` for skill fraud (Expert skills with 0 months duration), career start/end mismatches, and years of experience (YoE) inconsistencies. Successfully blocks exactly 68 fake candidates.
2. **Dense Semantic Retrieval:** Precomputes normalized dense embeddings locally using `all-MiniLM-L6-v2`. Computes cosine similarity scores at runtime in milliseconds using simple NumPy dot-products.
3. **JD Heuristic Multipliers:** Evaluates candidates on:
   - Target experience (5-9 years)
   - Hybrid office location (Pune/Noida preferred)
   - Career tenure stability (penalizes job-hoppers)
   - Domain alignment (penalizes consulting-only service backgrounds and computer vision-only backgrounds)
   - Availability (notice period buyouts)
   - Platform activity and response signals
4. **Factual Reasoning Generation:** Automatically writes a custom, non-templated 1-2 sentence description for each candidate summarizing their title, experience, company, location, skills, notice period, and match score.

---

## 📁 Repository Structure

```
india_runs/
├── India_runs_data_and_ai_challenge/   # Challenge datasets & schemas
├── models/
│   └── all-MiniLM-L6-v2/               # Local offline copy of model weights
├── utils/
│   ├── __init__.py
│   ├── honeypots.py                    # Honeypot checks
│   └── heuristics.py                   # Multipliers & heuristics
├── precompute.py                       # Embedding precomputation script
├── rank.py                             # Sandbox reproduction ranking script
├── app.py                              # Streamlit dashboard sandbox application
├── submission.csv                      # Final ranked output CSV (Top 100 shortlist)
├── submission_metadata.yaml            # Filled submission spec metadata
└── requirements.txt                    # Project dependencies
```

---

## 🛠️ Reproduction & Setup Instructions

### 1. Installation
Set up your virtual environment and install dependencies:
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Precomputation (One-time Offline Step)
Loads the model, downloads weights, processes all 100K profiles, and saves the compressed cache:
```bash
python precompute.py
```
*Note: The generated `precomputed_data.pkl.gz` (136 MB) is already included in this repository, so this step can be skipped.*

### 3. Generate Submission CSV (Runtime Evaluation Command)
Runs the offline ranker. It reads the candidate pool, looks up embeddings, runs the cascade logic, and saves the output in under 15 seconds:
```bash
python rank.py --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl --out ./submission.csv
```

### 4. Validate Submission CSV
Check format validity against rules:
```bash
python India_runs_data_and_ai_challenge/validate_submission.py submission.csv
```

---

## 📊 Streamlit Sandbox Application

To launch the interactive dashboard locally:
```bash
streamlit run app.py
```

The sandbox includes:
- ** Ranked Shortlist:** Beautiful table listing the top 100 candidates with download capability.
- ** Analytics Dashboard:** Plotly charts analyzing the experience, notice period, and location distribution of the shortlisted candidates.
- ** Profile Score Explainer:** Inspects individual profiles and breaks down their score into its semantic similarity and multiplier constituents for complete transparency.
- ** Integrity Report:** Logs all blocked honeypots and lists their fraudulent indicators.
