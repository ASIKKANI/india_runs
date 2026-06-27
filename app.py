import os
import json
import gzip
import pickle
import base64
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Import custom heuristics and honeypot checkers
from utils.honeypots import is_honeypot
from utils.heuristics import (
    check_service_only,
    calculate_tenure_multiplier,
    calculate_experience_multiplier,
    calculate_location_multiplier,
    calculate_behavioral_multiplier,
    check_cv_speech_bias
)
from rank import calculate_notice_multiplier, generate_candidate_reasoning, load_candidates

# Set page configuration
st.set_page_config(
    page_title="Redrob Candidate Discovery Engine",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Toggle State
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# Define dynamic theme colors
if IS_DARK:
    bg_color = "#09090b"
    bg_subtle = "#0c0c0f"
    card_bg = "#11131e"
    card_hover = "#171a2b"
    border_color = "#252b44"
    border_subtle = "#191d2e"
    text_color = "#fafafa"
    text_muted = "#a0aec0"
    text_dim = "#71717a"
    shadow = "none"
    accent = "#00f2fe"
    accent_gradient = "linear-gradient(135deg, #00f2fe 0%, #4facfe 100%)"
    card_grad = "linear-gradient(135deg, #151926 0%, #0f111a 100%)"
else:
    bg_color = "#f9fafb"
    bg_subtle = "#f3f4f6"
    card_bg = "#ffffff"
    card_hover = "#f9fafb"
    border_color = "#e5e7eb"
    border_subtle = "#f3f4f6"
    text_color = "#111827"
    text_muted = "#4b5563"
    text_dim = "#9ca3af"
    shadow = "0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.02)"
    accent = "#2563eb"
    accent_gradient = "linear-gradient(135deg, #2563eb 0%, #3b82f6 100%)"
    card_grad = "linear-gradient(135deg, #ffffff 0%, #f9fafb 100%)"

# Inject Dynamic Premium CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Outfit:wght@100..900&family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&display=swap');

/* Global Reset and Typography */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
    background-color: {bg_color} !important;
    color: {text_color} !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}

h1, h2, h3, h4, h5, h6 {{
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    color: {text_color} !important;
    letter-spacing: -0.02em !important;
}}

/* Hide default streamlit headers & footers for custom application branding */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton {{
    display: none !important;
}}

.block-container {{
    padding: 2rem 3rem !important;
    max-width: 1400px !important;
}}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {{
    background-color: {bg_subtle} !important;
    border-right: 1px solid {border_color} !important;
}}

[data-testid="stSidebar"] .block-container {{
    padding: 1.5rem 1rem !important;
}}

/* Custom Tabs styling */
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: {text_muted} !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    margin-right: 8px !important;
    transition: all 0.2s ease-in-out !important;
}}

button[data-baseweb="tab"]:hover {{
    color: {text_color} !important;
    background-color: {card_hover} !important;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: {accent if IS_DARK else '#ffffff'} !important;
    background: {card_bg if IS_DARK else accent} !important;
    border-color: {border_color if IS_DARK else accent} !important;
    box-shadow: {shadow} !important;
}}

[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
    display: none !important;
}}

[data-baseweb="tab-list"] {{
    gap: 0px !important;
    background: {bg_subtle} !important;
    border: 1px solid {border_color} !important;
    border-radius: 12px !important;
    padding: 4px !important;
    margin-bottom: 2rem !important;
}}

/* KPI metric cards */
.metric-card {{
    background: {card_grad};
    border: 1px solid {border_color};
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: {shadow};
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}}

.metric-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: {accent_gradient};
}}

.metric-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    border-color: {accent};
}}

.metric-label {{
    font-size: 0.85rem;
    color: {text_muted};
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}}

.metric-value {{
    font-size: 2.2rem;
    font-weight: 800;
    font-family: 'Outfit', sans-serif;
    background: {accent_gradient};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.04em;
    line-height: 1;
}}

.metric-card.blocked::before {{
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
}}
.metric-card.blocked .metric-value {{
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

/* Badges styling */
.badge {{
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}}
.badge-gold {{
    background: rgba(217, 119, 6, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(217, 119, 6, 0.3);
}}
.badge-silver {{
    background: rgba(156, 163, 175, 0.15);
    color: #9ca3af;
    border: 1px solid rgba(156, 163, 175, 0.3);
}}
.badge-bronze {{
    background: rgba(180, 83, 9, 0.15);
    color: #b45309;
    border: 1px solid rgba(180, 83, 9, 0.3);
}}
.badge-blue {{
    background: rgba(37, 99, 235, 0.1);
    color: #3b82f6;
    border: 1px solid rgba(37, 99, 235, 0.2);
}}
.badge-green {{
    background: rgba(34, 197, 94, 0.1);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.2);
}}
.badge-red {{
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.2);
}}

/* Custom Table / Cards for Shortlist */
.candidate-score-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: {accent};
    background: {bg_subtle};
    border: 1px solid {border_color};
    padding: 4px 12px;
    border-radius: 8px;
}}

/* Timeline Layout for experience */
.timeline {{
    position: relative;
    padding-left: 24px;
    border-left: 2px solid {border_color};
    margin-left: 10px;
    margin-top: 15px;
}}

.timeline-item {{
    position: relative;
    margin-bottom: 1.5rem;
}}

.timeline-item::before {{
    content: '';
    position: absolute;
    left: -31px;
    top: 4px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: {bg_color};
    border: 3px solid {accent};
}}

.timeline-item-title {{
    font-size: 0.95rem;
    font-weight: 700;
    color: {text_color};
    font-family: 'Outfit', sans-serif;
}}

.timeline-item-subtitle {{
    font-size: 0.8rem;
    color: {text_muted};
}}

/* Explainer list */
.explainer-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0rem;
    border-bottom: 1px solid {border_subtle};
}}
.explainer-row:last-child {{
    border-bottom: none;
}}
.explainer-label {{
    font-size: 0.85rem;
    color: {text_muted};
}}
.explainer-val {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.9rem;
    color: {text_color};
}}

/* Custom Header layout */
.header-container {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid {border_color};
    margin-bottom: 2rem;
}}

.logo-text {{
    font-size: 1.5rem;
    font-weight: 800;
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.03em;
    background: {accent_gradient};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.logo-text-subtitle {{
    font-size: 0.8rem;
    color: {text_muted};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-left: 2px;
}}

/* Custom file upload box wrapper styling override */
div[data-testid="stFileUploader"] {{
    background-color: {card_bg} !important;
    border: 2px dashed {border_color} !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: all 0.2s ease !important;
}}
div[data-testid="stFileUploader"]:hover {{
    border-color: {accent} !important;
}}

/* Info Box / Notification Cards */
.info-card {{
    background-color: {card_bg};
    border: 1px solid {border_color};
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: {shadow};
    margin-bottom: 1.5rem;
}}

.reasoning-box {{
    background: {bg_subtle};
    border-left: 4px solid {accent};
    border-radius: 4px;
    padding: 1rem;
    color: {text_color};
    font-style: italic;
    font-size: 0.95rem;
}}

/* Download Button styling */
div.stDownloadButton > button {{
    background: {accent_gradient} !important;
    color: white !important;
    border: none !important;
    padding: 0.6rem 1.5rem !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}}
div.stDownloadButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
    opacity: 0.95 !important;
}}

/* Chart styling container */
.chart-wrap {{
    background: {card_bg};
    border: 1px solid {border_color};
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: {shadow};
    margin-bottom: 1.5rem;
}}
.chart-title {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {text_color};
    font-family: 'Outfit', sans-serif;
}}
.chart-subtitle {{
    font-size: 0.8rem;
    color: {text_dim};
    margin-bottom: 1.25rem;
}}
</style>
""", unsafe_allow_html=True)

# Helper to load models
@st.cache_resource
def load_local_model():
    local_model_path = "./models/all-MiniLM-L6-v2"
    if os.path.exists(local_model_path):
        return SentenceTransformer(local_model_path)
    else:
        return SentenceTransformer('all-MiniLM-L6-v2')

# Helper to load precomputed data
@st.cache_data
def load_precomputed_cache():
    precomputed_cache_path = "./precomputed_data.pkl.gz"
    if os.path.exists(precomputed_cache_path):
        with gzip.open(precomputed_cache_path, "rb") as f:
            data = pickle.load(f)
            # Create a fast lookup map for candidate_id to index
            if data and "candidate_ids" in data:
                data["id_to_idx"] = {cid: idx for idx, cid in enumerate(data["candidate_ids"])}
            return data
    return None

def score_candidate(cand, query_emb, model, precomputed):
    cid = cand['candidate_id']
    profile = cand.get('profile', {})
    career = cand.get('career_history', [])
    skills = cand.get('skills', [])
    signals = cand.get('redrob_signals', {})
    
    # Honeypot detection
    is_hp = False
    hp_reason = ""
    if precomputed and cid in precomputed["honeypots"]:
        is_hp, hp_reason = precomputed["honeypots"][cid]
    else:
        is_hp, hp_reason = is_honeypot(cand)
        
    if is_hp:
        return {
            "candidate_id": cid,
            "name": profile.get("anonymized_name", "Anonymized"),
            "title": profile.get("current_title", "Engineer"),
            "company": profile.get("current_company", "N/A"),
            "score": 0.0,
            "similarity": 0.0,
            "exp_mult": 0.0,
            "loc_mult": 0.0,
            "tenure_mult": 0.0,
            "behav_mult": 0.0,
            "notice_mult": 0.0,
            "service_mult": 0.0,
            "cv_mult": 0.0,
            "is_honeypot": True,
            "hp_reason": hp_reason,
            "reasoning": f"Profile flagged due to data consistency issues: {hp_reason}"
        }
        
    # Semantic Similarity lookup or compute
    similarity_score = 0.0
    if precomputed and "id_to_idx" in precomputed and cid in precomputed["id_to_idx"]:
        idx = precomputed["id_to_idx"][cid]
        cand_emb = precomputed["embeddings"][idx]
        similarity_score = float(np.dot(cand_emb, query_emb))
    else:
        text_rep = f"Title: {profile.get('current_title', '')}. Headline: {profile.get('headline', '')}. "
        top_skills = [s.get('name', '') for s in skills if s.get('name')]
        skills_str = ", ".join(top_skills[:15])
        if skills_str:
            text_rep += f"Skills: {skills_str}. "
        roles = ", ".join([job.get('title', '') for job in career if job.get('title')])
        if roles:
            text_rep += f"Roles: {roles}."
        text_rep = text_rep[:400]
        cand_emb = model.encode([text_rep], normalize_embeddings=True, convert_to_numpy=True)[0]
        similarity_score = float(np.dot(cand_emb, query_emb))
        
    similarity_score = max(0.0, similarity_score)
    
    # Calculate multipliers
    exp_mult = calculate_experience_multiplier(profile.get('years_of_experience', 0))
    loc_mult = calculate_location_multiplier(
        profile.get('location', ''), 
        profile.get('country', ''), 
        signals.get('willing_to_relocate', False)
    )
    tenure_mult = calculate_tenure_multiplier(career)
    behav_mult = calculate_behavioral_multiplier(signals)
    notice_mult = calculate_notice_multiplier(signals.get('notice_period_days', 30))
    
    # Penalties
    service_mult = 0.1 if check_service_only(career) else 1.0
    cv_mult = 0.3 if check_cv_speech_bias(skills) else 1.0
    
    # Score calculation
    final_score = similarity_score * exp_mult * loc_mult * tenure_mult * behav_mult * notice_mult * service_mult * cv_mult
    final_score_scaled = round(final_score * 100.0, 4)
    
    reasoning = generate_candidate_reasoning(cand, final_score_scaled)
    
    return {
        "candidate_id": cid,
        "name": profile.get("anonymized_name", "Anonymized"),
        "title": profile.get("current_title", "Engineer"),
        "company": profile.get("current_company", "N/A"),
        "score": final_score_scaled,
        "similarity": round(similarity_score, 4),
        "exp_mult": round(exp_mult, 2),
        "loc_mult": round(loc_mult, 2),
        "tenure_mult": round(tenure_mult, 2),
        "behav_mult": round(behav_mult, 2),
        "notice_mult": round(notice_mult, 2),
        "service_mult": round(service_mult, 2),
        "cv_mult": round(cv_mult, 2),
        "is_honeypot": False,
        "hp_reason": "",
        "reasoning": reasoning
    }

def main():
    # Setup custom brand header
    header_col1, header_col2 = st.columns([8, 2])
    with header_col1:
        logo_html = ""
        if os.path.exists("redrob_logo.png"):
            try:
                with open("redrob_logo.png", "rb") as image_file:
                    encoded_logo = base64.b64encode(image_file.read()).decode()
                logo_html = f'<img src="data:image/png;base64,{encoded_logo}" style="height: 38px; margin-right: 12px; vertical-align: middle;"/>'
            except Exception:
                pass
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; padding-top: 0.5rem; padding-bottom: 0.5rem;">
            {logo_html}
            <div>
                <span class="logo-text">Redrob AI Ranker</span>
                <span class="logo-text-subtitle">Interactive Candidate Discovery Sandbox</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with header_col2:
        theme_label = "☀️ Light View" if IS_DARK else "🌙 Dark View"
        st.button(theme_label, on_click=toggle_theme, use_container_width=True)

    # Sidebar setup
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload candidates dataset (.json, .jsonl)", type=["json", "jsonl"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Target Role Requirements")
    st.sidebar.info("""
    **Senior AI Engineer — Founding Team**
    * **Experience:** 5-9 years (preferred)
    * **Location:** Pune / Noida, India (Hybrid)
    * **Core Tech:** Dense embeddings retrieval, vector DBs (FAISS, Milvus, Qdrant, Pinecone), Python, ranking evaluation frameworks (NDCG, MRR).
    """)
    
    # Load Model & Cache
    model = load_local_model()
    precomputed = load_precomputed_cache()
    
    # Query definition
    jd_query = (
        "Senior AI Engineer, Applied Machine Learning, NLP, Search, Information Retrieval, Ranking. "
        "Founding Team. Experience: 5-9 years. Technologies: Sentence Transformers, Embeddings, "
        "Vector Databases (Pinecone, Qdrant, Milvus, FAISS, Weaviate, OpenSearch), Python, LLMs, "
        "Fine-tuning (LoRA, QLoRA, PEFT), Evaluation frameworks (NDCG, MRR). Product company background."
    )
    query_emb = model.encode([jd_query], normalize_embeddings=True, convert_to_numpy=True)[0]
    
    # Plotly Chart Theming Base
    PLOT_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=11),
        margin=dict(l=30, r=20, t=40, b=35),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.05)",
            zerolinecolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.05)",
            tickfont=dict(size=10, color="#7b7b85" if not IS_DARK else "#a1a1aa"),
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.05)",
            zerolinecolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.05)",
            tickfont=dict(size=10, color="#7b7b85" if not IS_DARK else "#a1a1aa"),
        ),
    )
    
    # Process uploaded data
    if uploaded_file is not None:
        # Load profiles
        candidates = []
        try:
            content = uploaded_file.read().decode("utf-8")
            # Try loading as JSON array
            try:
                candidates = json.loads(content)
            except json.JSONDecodeError:
                # Try loading as JSON Lines
                candidates = [json.loads(line) for line in content.splitlines() if line.strip()]
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            return
            
        total_candidates = len(candidates)
        
        # Run scoring
        with st.spinner("Scoring candidates in real-time..."):
            scored_candidates = [score_candidate(cand, query_emb, model, precomputed) for cand in candidates]
            
        # Create DataFrames
        df = pd.DataFrame(scored_candidates)
        
        # Honeypot filtering
        honeypots_df = df[df["is_honeypot"] == True]
        valid_df = df[df["is_honeypot"] == False]
        
        # Sort and Rank valid
        valid_df = valid_df.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
        valid_df["rank"] = valid_df.index + 1
        
        # Stats metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Pool Size</div><div class="metric-value">{total_candidates}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card blocked"><div class="metric-label">Honeypots Blocked</div><div class="metric-value">{len(honeypots_df)}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">High-Fit Candidates (Score >= 40)</div><div class="metric-value">{len(valid_df[valid_df["score"] >= 40.0])}</div></div>', unsafe_allow_html=True)
        with col4:
            top_val = valid_df["score"].max() if len(valid_df) > 0 else 0.0
            st.markdown(f'<div class="metric-card"><div class="metric-label">Top Fit Score</div><div class="metric-value">{top_val:.2f}</div></div>', unsafe_allow_html=True)
            
        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranked Shortlist", "📊 Analytics Dashboard", "🔍 Candidate Profile Inspector", "🛡️ Integrity & Honeypot Report"])
        
        with tab1:
            top_100 = valid_df.head(100)
            
            # Show Top 3 match cards
            st.markdown("### 🥇 Top 3 Candidate Matches")
            top_3 = top_100.head(3)
            t3_cols = st.columns(3)
            
            badges = ["badge-gold", "badge-silver", "badge-bronze"]
            medal_icons = ["🥇", "🥈", "🥉"]
            borders = [
                f"border: 1px solid rgba(217, 119, 6, 0.4); box-shadow: 0 0 12px rgba(217, 119, 6, 0.12);",
                f"border: 1px solid rgba(156, 163, 175, 0.4); box-shadow: 0 0 12px rgba(156, 163, 175, 0.12);",
                f"border: 1px solid rgba(180, 83, 9, 0.4); box-shadow: 0 0 12px rgba(180, 83, 9, 0.12);"
            ]
            
            for idx, (_, row) in enumerate(top_3.iterrows()):
                with t3_cols[idx]:
                    st.markdown(f"""
                    <div class="metric-card" style="{borders[idx]}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span class="badge {badges[idx]}">{medal_icons[idx]} Rank {row['rank']}</span>
                            <span class="candidate-score-badge">{row['score']:.2f}</span>
                        </div>
                        <div style="font-size: 1.25rem; font-weight: 700; font-family: 'Outfit'; color: {text_color}; margin-top: 0.5rem;">{row['name']}</div>
                        <div style="font-size: 0.9rem; color: {text_muted}; margin-bottom: 0.75rem;">{row['title']} at <b>{row['company']}</b></div>
                        <div class="reasoning-box" style="font-size: 0.82rem; margin-top: 0.5rem;">
                            {row['reasoning']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("### Top 100 Shortlist Registry")
            
            # Show table
            st.dataframe(
                top_100[["rank", "candidate_id", "name", "title", "company", "score", "reasoning"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Download submission CSV button
            csv_data = top_100[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False)
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Download submission.csv",
                data=csv_data,
                file_name="submission.csv",
                mime="text/csv"
            )
            
        with tab2:
            st.markdown("### Shortlist Feature Distributions")
            top_100_full = []
            top_100_ids = set(top_100["candidate_id"])
            for cand in candidates:
                if cand["candidate_id"] in top_100_ids:
                    top_100_full.append(cand)
                    
            if top_100_full:
                # Experience Distribution
                yoe_list = [c["profile"]["years_of_experience"] for c in top_100_full]
                fig1 = px.histogram(
                    x=yoe_list, 
                    nbins=10, 
                    labels={'x': 'Years of Experience', 'y': 'Count'}, 
                    color_discrete_sequence=[accent]
                )
                fig1.update_layout(
                    PLOT_LAYOUT,
                    title=dict(text="Experience Distribution (Target: 5-9 years)", font=dict(family="Outfit", size=14, color=text_color)),
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                
                # Location Distribution
                locs = [c["profile"]["location"] for c in top_100_full]
                fig2 = px.histogram(
                    y=locs, 
                    labels={'y': 'Location', 'x': 'Count'}, 
                    color_discrete_sequence=[accent]
                )
                fig2.update_layout(
                    PLOT_LAYOUT,
                    title=dict(text="Geographic Location Breakdown", font=dict(family="Outfit", size=14, color=text_color)),
                    margin=dict(l=100, r=20, t=50, b=40),
                )
                
                # Notice period distribution
                notice_periods = [c["redrob_signals"]["notice_period_days"] for c in top_100_full]
                fig3 = px.histogram(
                    x=notice_periods, 
                    labels={'x': 'Notice Period (Days)', 'y': 'Count'}, 
                    color_discrete_sequence=[accent]
                )
                fig3.update_layout(
                    PLOT_LAYOUT,
                    title=dict(text="Notice Period Distribution", font=dict(family="Outfit", size=14, color=text_color)),
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="chart-wrap">
                        <div class="chart-title">Experience Distribution</div>
                        <div class="chart-subtitle">Distribution of years of experience in shortlist (Target: 5-9 YoE)</div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="chart-wrap">
                        <div class="chart-title">Notice Period</div>
                        <div class="chart-subtitle">Recruiter availability distribution (Days to hire)</div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="chart-wrap">
                        <div class="chart-title">Location Distribution</div>
                        <div class="chart-subtitle">Breakdown of geographic locations of candidate pool</div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No candidates scored high enough to generate distributions.")
                
        with tab3:
            st.markdown("### Profile Score Explainer")
            selected_id = st.selectbox("Select a candidate to inspect:", options=top_100["candidate_id"].tolist())
            
            if selected_id:
                cand_info = next(c for c in candidates if c["candidate_id"] == selected_id)
                cand_score_row = top_100[top_100["candidate_id"] == selected_id].iloc[0]
                
                # Render Timeline HTML
                timeline_html = '<div class="timeline">'
                for job in cand_info.get('career_history', []):
                    dur_str = f"{job.get('duration_months', 0)} mos" if job.get('duration_months') else ""
                    dates = f"{job.get('start_date')} → {job.get('end_date') or 'Present'}"
                    timeline_html += f"""
                    <div class="timeline-item">
                        <div class="timeline-item-title">{job.get('title')}</div>
                        <div class="timeline-item-subtitle"><b>{job.get('company')}</b> | {dates} ({dur_str})</div>
                    </div>
                    """
                timeline_html += '</div>'
                
                # Render Skills HTML
                skills_html = ""
                for s in cand_info.get('skills', []):
                    prof = s.get('proficiency', 'Intermediate')
                    skills_html += f'<span class="badge badge-blue" style="margin-right: 6px; margin-bottom: 6px;">{s.get("name")} ({prof})</span>'
                
                # Helper function to format multiplier badge
                def get_mult_badge(val):
                    if val > 1.0:
                        return f'<span class="badge badge-green" style="font-family:JetBrains Mono;">{val:.2f}x</span>'
                    elif val == 1.0:
                        return f'<span class="badge badge-silver" style="font-family:JetBrains Mono;">{val:.2f}x</span>'
                    elif val >= 0.7:
                        return f'<span class="badge badge-amber" style="font-family:JetBrains Mono;">{val:.2f}x</span>'
                    else:
                        return f'<span class="badge badge-red" style="font-family:JetBrains Mono;">{val:.2f}x</span>'
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"""<div class="info-card">
<h3 style="margin-top:0;">👤 {cand_info['profile']['anonymized_name']}</h3>
<p style="font-size:1.15rem; font-style:italic; color:{text_muted}; margin-bottom: 1.25rem;">{cand_info['profile']['headline']}</p>
<div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:1.5rem;">
<span class="badge badge-green">💼 {cand_info['profile']['current_title']}</span>
<span class="badge badge-blue">🏢 {cand_info['profile']['current_company']}</span>
<span class="badge badge-gold">📍 {cand_info['profile']['location']}, {cand_info['profile']['country']}</span>
<span class="badge badge-silver">📅 {cand_info['profile']['years_of_experience']} YoE</span>
</div>
<h4 style="margin-top: 1.5rem; margin-bottom: 0.5rem;">💼 Career Experience</h4>
{timeline_html}
<h4 style="margin-top: 2rem; margin-bottom: 0.75rem;">⚙️ Core Skills</h4>
<div style="display:flex; flex-wrap:wrap;">
{skills_html}
</div>
</div>""", unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""<div class="info-card">
<h3 style="margin-top:0;">⚡ Scoring Cascade Breakdown</h3>
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2rem; background:{bg_subtle}; padding: 0.75rem 1.25rem; border-radius: 8px; border:1px solid {border_color};">
<span style="font-size:1.05rem; font-weight:700; color:{text_color};">Cascade Match Score</span>
<span class="candidate-score-badge" style="font-size:1.6rem; padding:4px 14px;">{cand_score_row['score']:.2f}</span>
</div>
<div class="explainer-row">
<span class="explainer-label"><b>Base Semantic Similarity</b><br><small style="color:{text_dim};">Embeddings title & headline relevance</small></span>
<span class="explainer-val">{cand_score_row['similarity']:.4f}</span>
</div>
<div class="explainer-row">
<span class="explainer-label"><b>Experience Fit Multiplier</b><br><small style="color:{text_dim};">Optimal range: 5 to 9 years of experience</small></span>
<span class="explainer-val">{get_mult_badge(cand_score_row['exp_mult'])}</span>
</div>
<div class="explainer-row">
<span class="explainer-label"><b>Location Fit Multiplier</b><br><small style="color:{text_dim};">Noida/Pune location or relocation willingness</small></span>
<span class="explainer-val">{get_mult_badge(cand_score_row['loc_mult'])}</span>
</div>
<div class="explainer-row">
<span class="explainer-label"><b>Tenure Stability Multiplier</b><br><small style="color:{text_dim};">Avg job duration (catches frequent job-hopping)</small></span>
<span class="explainer-val">{get_mult_badge(cand_score_row['tenure_mult'])}</span>
</div>
<div class="explainer-row">
<span class="explainer-label"><b>Notice Period Multiplier</b><br><small style="color:{text_dim};">Notice period timeline (boost for immediate start)</small></span>
<span class="explainer-val">{get_mult_badge(cand_score_row['notice_mult'])}</span>
</div>
<div class="explainer-row">
<span class="explainer-label"><b>Active Platform Signals</b><br><small style="color:{text_dim};">Recruiter response rate & platform activity logs</small></span>
<span class="explainer-val">{get_mult_badge(cand_score_row['behav_mult'])}</span>
</div>
<div class="explainer-row">
<span class="explainer-label"><b>Service Only Penalty</b><br><small style="color:{text_dim};">Applies 0.1x penalty for pure IT outsourcing backgrounds</small></span>
<span class="explainer-val">{get_mult_badge(cand_score_row['service_mult'])}</span>
</div>
<div class="explainer-row">
<span class="explainer-label"><b>Computer Vision Bias Penalty</b><br><small style="color:{text_dim};">Applies 0.3x penalty for CV/speech heavy profiles</small></span>
<span class="explainer-val">{get_mult_badge(cand_score_row['cv_mult'])}</span>
</div>
<h4 style="margin-top:2rem; margin-bottom:0.75rem;">🧠 AI Recruiter Reasoning</h4>
<div class="reasoning-box">
{cand_score_row['reasoning']}
</div>
</div>""", unsafe_allow_html=True)
                    
        with tab4:
            st.markdown("### 🛡️ Integrity & Fraud Detection Audit")
            st.markdown(f"""<div class="info-card" style="border-left: 5px solid #ef4444; background-color: {card_bg}; margin-bottom: 1.5rem;">
<h4 style="margin-top:0; color: #ef4444; font-family: Outfit;">System Integrity Verification Summary</h4>
<p style="color: {text_muted}; margin-bottom: 0.75rem;">
The ranking engine executes 5 strict data consistency checkers on candidate history to filter out inflated or fake profiles.
Flagged profiles are immediately quarantine-listed with a fit score of <b>0.00</b>.
</p>
<div style="font-size: 1.15rem; font-weight: 700; color: #ef4444;">
⚠️ {len(honeypots_df)} Honeypots Quarantined and Filtered Out
</div>
</div>""", unsafe_allow_html=True)
            
            if len(honeypots_df) > 0:
                st.dataframe(
                    honeypots_df[["candidate_id", "name", "title", "company", "hp_reason"]],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("🎉 No honeypots detected in the uploaded dataset!")
                
    else:
        # Hero Section
        st.markdown(f"""
        <div class="info-card" style="padding: 3rem 2.5rem; text-align: center; background: {card_grad}; margin-bottom: 2rem; border-color: {border_color};">
            <h1 style="font-size: 2.6rem; margin-top: 0; background: {accent_gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: Outfit; font-weight: 800;">
                Redrob Candidate Discover Engine
            </h1>
            <p style="font-size: 1.15rem; color: {text_muted}; max-width: 800px; margin: 0 auto 1.5rem; line-height: 1.5;">
                A state-of-the-art Multi-Stage Cascade Ranking sandbox. Our engine scans profiles for employment fraud, computes semantic relevance weights against targets, and scores talent using custom multipliers in real-time.
            </p>
            <div style="display: inline-flex; align-items: center; gap: 10px; padding: 8px 18px; background: {bg_subtle}; border: 1px solid {border_color}; border-radius: 20px; font-size: 0.9rem; color: {text_color}; font-weight: 600;">
                👈 Upload a candidate dataset file (.json or .jsonl) in the sidebar to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if precomputed:
            st.markdown(f"""<div class="info-card" style="border-color: {border_color};">
<h3 style="margin-top:0; font-family: Outfit; font-weight: 700;">🛡️ Cascade Ranking Architecture Blueprint</h3>
<p style="color: {text_muted}; margin-bottom: 1.5rem;">The ranking sandbox runs the following pipeline layers to identify and rank candidate matches:</p>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.25rem;">
<div class="metric-card" style="padding: 1.2rem; background: {card_bg};">
<h5 style="margin-top: 0; color: #ef4444; font-family: Outfit; font-weight: 700;">1. Integrity Scan</h5>
<p style="font-size: 0.8rem; color: {text_muted}; margin: 0; line-height: 1.4;">Checks for date anomalies, skill duration fabrication, and job overlap fraud. Flags and removes candidates instantly.</p>
</div>
<div class="metric-card" style="padding: 1.2rem; background: {card_bg};">
<h5 style="margin-top: 0; color: #3b82f6; font-family: Outfit; font-weight: 700;">2. Dense Retrieval</h5>
<p style="font-size: 0.8rem; color: {text_muted}; margin: 0; line-height: 1.4;">Computes semantic cosine similarity of title, summary, and skills against JD using local <code>all-MiniLM-L6-v2</code>.</p>
</div>
<div class="metric-card" style="padding: 1.2rem; background: {card_bg};">
<h5 style="margin-top: 0; color: #22c55e; font-family: Outfit; font-weight: 700;">3. Multipliers & Penalties</h5>
<p style="font-size: 0.8rem; color: {text_muted}; margin: 0; line-height: 1.4;">Adjusts candidate ranks using experience, stability, notice period availability, and specific domain-fit indicators.</p>
</div>
<div class="metric-card" style="padding: 1.2rem; background: {card_bg};">
<h5 style="margin-top: 0; color: #a855f7; font-family: Outfit; font-weight: 700;">4. Factual Reasoning</h5>
<p style="font-size: 0.8rem; color: {text_muted}; margin: 0; line-height: 1.4;">Formulates natural, non-templated summaries of experience and fit metrics to justify final candidate rank.</p>
</div>
</div>
</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
