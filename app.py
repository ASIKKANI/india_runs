import os
import json
import gzip
import pickle
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
    page_title="Redrob Candidate Ranking Engine",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stApp {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #00f2fe;
        font-family: 'Outfit', sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2640 0%, #0f1526 100%);
        border: 1px solid #2e3a59;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #00f2fe;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0;
    }
    .candidate-card {
        background: #1a2035;
        border-left: 5px solid #00f2fe;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        color: #ffffff;
    }
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
    st.title("🏆 Redrob Intelligent Candidate Discovery")
    st.subheader("Interactive Cascade Ranking Engine Sandbox")
    
    # Sidebar
    if os.path.exists("redrob_logo.png"):
        st.sidebar.image("redrob_logo.png", width=150)
    else:
        st.sidebar.title("Redrob AI")
    st.sidebar.header("📁 Data & Controls")
    
    # Upload file
    uploaded_file = st.sidebar.file_uploader("Upload candidates dataset (.json, .jsonl)", type=["json", "jsonl"])
    
    # Job Description in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 target Role Requirements")
    st.sidebar.info("""
    **Senior AI Engineer — Founding Team**
    - **Experience:** 5-9 years (preferred)
    - **Location:** Pune / Noida, India (Hybrid)
    - **Core Tech:** Dense embeddings retrieval, vector DBs (FAISS, Milvus, Qdrant, Pinecone), Python, ranking evaluation frameworks (NDCG, MRR).
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
    
    # Process uploaded data
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        
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
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_candidates}</div><div class="metric-label">Total Pool Size</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(honeypots_df)}</div><div class="metric-label">Honeypots Blocked</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(valid_df[valid_df["score"] >= 40.0])}</div><div class="metric-label">High-Fit Candidates (Score >= 40)</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{valid_df["score"].max() if len(valid_df) > 0 else 0.0:.2f}</div><div class="metric-label">Top Fit Score</div></div>', unsafe_allow_html=True)
            
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranked Shortlist", "📊 Analytics Dashboard", "🔍 Candidate Profile Inspector", "🛡️ Integrity & Honeypot Report"])
        
        with tab1:
            st.markdown("### Top 100 Candidates Shortlist")
            top_100 = valid_df.head(100)
            
            # Show table
            st.dataframe(
                top_100[["rank", "candidate_id", "name", "title", "company", "score", "reasoning"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Download submission CSV button
            csv_data = top_100[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False)
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
                    # Resolve full details
                    top_100_full.append(cand)
                    
            if top_100_full:
                # Experience Distribution
                yoe_list = [c["profile"]["years_of_experience"] for c in top_100_full]
                fig1 = px.histogram(
                    x=yoe_list, 
                    nbins=10, 
                    labels={'x': 'Years of Experience'}, 
                    title="Experience Distribution (Target: 5-9 years)",
                    color_discrete_sequence=['#00f2fe']
                )
                
                # Location Distribution
                locs = [c["profile"]["location"] for c in top_100_full]
                fig2 = px.histogram(
                    y=locs, 
                    labels={'y': 'Location'}, 
                    title="Geographic Location Breakdown",
                    color_discrete_sequence=['#4facfe']
                )
                
                # Notice period distribution
                notice_periods = [c["redrob_signals"]["notice_period_days"] for c in top_100_full]
                fig3 = px.histogram(
                    x=notice_periods, 
                    labels={'x': 'Notice Period (Days)'}, 
                    title="Notice Period Distribution",
                    color_discrete_sequence=['#00f2fe']
                )
                
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig3, use_container_width=True)
                with c2:
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No candidates scored high enough to generate distributions.")
                
        with tab3:
            st.markdown("### Profile Score Explainer")
            selected_id = st.selectbox("Select a candidate to inspect:", options=top_100["candidate_id"].tolist())
            
            if selected_id:
                cand_info = next(c for c in candidates if c["candidate_id"] == selected_id)
                cand_score_row = top_100[top_100["candidate_id"] == selected_id].iloc[0]
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"#### 👤 {cand_info['profile']['anonymized_name']}")
                    st.markdown(f"**Headline:** {cand_info['profile']['headline']}")
                    st.markdown(f"**Current Title:** {cand_info['profile']['current_title']} at **{cand_info['profile']['current_company']}** ({cand_info['profile']['current_company_size']} employees)")
                    st.markdown(f"**Years of Experience:** {cand_info['profile']['years_of_experience']} years")
                    st.markdown(f"**Location:** {cand_info['profile']['location']}, {cand_info['profile']['country']}")
                    
                    st.markdown("##### 💼 Career History")
                    for job in cand_info.get('career_history', []):
                        st.markdown(f"- **{job.get('title')}** at *{job.get('company')}* ({job.get('start_date')} -> {job.get('end_date') or 'Present'}) : {job.get('duration_months')} months")
                        
                    st.markdown("##### ⚙️ Skills Inventory")
                    st.write(", ".join([f"{s['name']} ({s['proficiency']})" for s in cand_info.get('skills', [])]))
                    
                with col2:
                    st.markdown("#### ⚡ Scoring Cascade Breakdown")
                    st.metric(label="Final Score", value=f"{cand_score_row['score']:.2f}")
                    
                    # Explainer multipliers
                    st.markdown(f"**Base Semantic Similarity:** `{cand_score_row['similarity']:.4f}`")
                    st.markdown(f"**Experience Fit Multiplier:** `{cand_score_row['exp_mult']}x` (Target 5-9 YoE)")
                    st.markdown(f"**Location Fit Multiplier:** `{cand_score_row['loc_mult']}x` (Target Noida/Pune)")
                    st.markdown(f"**Tenure Multiplier (Job-Hopping Penalty):** `{cand_score_row['tenure_mult']}x` (Target >18 mos average)")
                    st.markdown(f"**Notice Period Multiplier:** `{cand_score_row['notice_mult']}x` (Target sub-30 days)")
                    st.markdown(f"**Active Platform Signals:** `{cand_score_row['behav_mult']}x` (Based on response rates, last active)")
                    st.markdown(f"**Consulting/Service Only Penalty:** `{cand_score_row['service_mult']}x` (Target product company background)")
                    st.markdown(f"**Computer Vision Bias Penalty:** `{cand_score_row['cv_mult']}x` (Target NLP/IR background)")
                    
                    st.markdown("##### 🧠 AI Recruiter Reasoning")
                    st.info(cand_score_row['reasoning'])
                    
        with tab4:
            st.markdown("### Blocked Honeypot Candidates")
            st.write(f"The system detected and blocked **{len(honeypots_df)}** profiles that contained data inconsistencies or fraudulent claims.")
            
            if len(honeypots_df) > 0:
                st.dataframe(
                    honeypots_df[["candidate_id", "name", "title", "company", "hp_reason"]],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("No honeypots detected in the uploaded dataset!")
                
    else:
        # File upload prompt
        st.info("👈 Please upload a candidate JSON/JSONL dataset file in the sidebar to begin ranking!")
        
        # Show default/sample state if precomputed cache is loaded
        if precomputed:
            st.success("Precomputed offline embedding cache is loaded and ready.")
            st.markdown("""
            ### Engine Architecture Blueprint
            The Candidate Ranking Engine implements a **Multi-Stage Cascade Ranking Architecture**:
            1. **Stage 1 (Anomaly Detection):** Instantly filters out candidate profiles with date mismatches, skill duration fraud, or overlapping careers (forces final score to `0.0`).
            2. **Stage 2 (Dense Retrieval):** Uses a local, offline `SentenceTransformer('all-MiniLM-L6-v2')` model to match the candidate's titles, headline, and top skills against the founding AI engineer requirements.
            3. **Stage 3 (Heuristic Multipliers):** Re-ranks candidates by incorporating experience fit, tenure stability, notice period buyouts, target location matches, and recruiter response metrics.
            4. **Stage 4 (Explainable Output):** Formulates candidate-specific factual reasonings and outputs a structured sorting with strict tie-breakers.
            """)

if __name__ == "__main__":
    main()
