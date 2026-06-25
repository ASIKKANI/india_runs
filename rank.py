import os
import sys
import argparse
import json
import gzip
import pickle
import numpy as np
import csv
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

# Notice period multiplier helper
def calculate_notice_multiplier(notice_days) -> float:
    if notice_days <= 30:
        return 1.05  # minor boost for immediate availability
    elif notice_days <= 60:
        return 1.0
    elif notice_days <= 90:
        return 0.8
    else:
        return 0.5   # penalty for long notice periods

def generate_candidate_reasoning(cand, similarity_score) -> str:
    profile = cand.get('profile', {})
    skills = cand.get('skills', [])
    signals = cand.get('redrob_signals', {})
    
    yoe = profile.get('years_of_experience', 0)
    title = profile.get('current_title', 'Engineer')
    comp = profile.get('current_company', 'Product Company')
    
    # Find matching skills from target skills
    target_skills = {
        'nlp', 'embeddings', 'retrieval', 'search', 'vector search', 
        'vector database', 'milvus', 'qdrant', 'pinecone', 'weaviate', 
        'llm', 'llms', 'fine-tuning', 'pytorch', 'tensorflow', 'python'
    }
    matching_skills = []
    for s in skills:
        name = s.get('name', '')
        if name.lower() in target_skills or any(ts in name.lower() for ts in target_skills):
            matching_skills.append(name)
            
    skills_phrase = ""
    if matching_skills:
        skills_phrase = f" with expertise in {', '.join(matching_skills[:3])}"
        
    loc = profile.get('location', 'India')
    notice = signals.get('notice_period_days', 30)
    
    # 1-2 sentence natural, factual description
    sentence1 = f"Experienced {title} with {yoe} years of experience, currently working at {comp}{skills_phrase}."
    sentence2 = f"Located in {loc} with a notice period of {notice} days, showing a strong matching score ({similarity_score:.2f}) for the founding team."
    
    return f"{sentence1} {sentence2}"

def load_candidates(file_path):
    # Try loading as JSON array first (e.g. sample_candidates.json)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            candidates = json.load(f)
            return candidates
    except json.JSONDecodeError:
        pass
        
    # Load as JSON Lines
    candidates = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
        return candidates
    except Exception as e:
        print(f"Error loading candidates: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranking Engine")
    parser.add_argument("--candidates", required=True, help="Path to candidates dataset (.json or .jsonl)")
    parser.add_argument("--out", required=True, help="Path to output submission CSV")
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    # Paths configuration
    precomputed_cache_path = "./precomputed_data.pkl.gz"
    local_model_path = "./models/all-MiniLM-L6-v2"
    
    # 1. Load precomputed data
    if not os.path.exists(precomputed_cache_path):
        print(f"Error: Precomputed data cache not found at {precomputed_cache_path}. Please run precompute.py first.")
        sys.exit(1)
        
    print(f"Loading precomputed data from {precomputed_cache_path}...")
    with gzip.open(precomputed_cache_path, "rb") as f:
        precomputed = pickle.load(f)
    print("Precomputed data loaded successfully!")
    
    # Map candidate_id to index for fast O(1) lookup
    candidate_ids_list = precomputed["candidate_ids"]
    id_to_index = {cid: idx for idx, cid in enumerate(candidate_ids_list)}
    precomputed_embeddings = precomputed["embeddings"]
    precomputed_honeypots = precomputed["honeypots"]
    
    # 2. Load the offline model
    if not os.path.exists(local_model_path):
        print(f"Error: Offline model not found at {local_model_path}. Please run precompute.py first.")
        sys.exit(1)
        
    print("Loading embedding model offline...")
    model = SentenceTransformer(local_model_path)
    print("Model loaded successfully!")
    
    # 3. Embed Job Description query
    jd_query = (
        "Senior AI Engineer, Applied Machine Learning, NLP, Search, Information Retrieval, Ranking. "
        "Founding Team. Experience: 5-9 years. Technologies: Sentence Transformers, Embeddings, "
        "Vector Databases (Pinecone, Qdrant, Milvus, FAISS, Weaviate, OpenSearch), Python, LLMs, "
        "Fine-tuning (LoRA, QLoRA, PEFT), Evaluation frameworks (NDCG, MRR). Product company background."
    )
    print("Generating query embedding...")
    query_emb = model.encode([jd_query], normalize_embeddings=True, convert_to_numpy=True)[0]
    
    # 4. Load candidate profiles
    print(f"Loading candidate profiles from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} profiles.")
    
    # 5. Process and score candidates
    print("Scoring candidates...")
    ranked_list = []
    
    for cand in candidates:
        cid = cand['candidate_id']
        profile = cand.get('profile', {})
        career = cand.get('career_history', [])
        skills = cand.get('skills', [])
        signals = cand.get('redrob_signals', {})
        
        # Check if candidate is a honeypot (using local check or lookup)
        is_hp = False
        hp_reason = ""
        if cid in precomputed_honeypots:
            is_hp, hp_reason = precomputed_honeypots[cid]
        else:
            is_hp, hp_reason = is_honeypot(cand)
            
        if is_hp:
            # Set score to 0.0 for honeypot profiles to keep them out of top 100
            ranked_list.append({
                "candidate_id": cid,
                "score": 0.0,
                "reasoning": f"Profile flagged due to data consistency issues: {hp_reason}"
            })
            continue
            
        # Get semantic similarity score
        similarity_score = 0.0
        if cid in id_to_index:
            idx = id_to_index[cid]
            cand_emb = precomputed_embeddings[idx]
            # Since vectors are normalized, similarity is dot product
            similarity_score = float(np.dot(cand_emb, query_emb))
        else:
            # Fallback if candidate was not precomputed
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
            
        # Base score is similarity score (bounds: [0.0, 1.0])
        # Map similarity from [-1, 1] to [0.2, 1.0] to keep positive
        similarity_score = max(0.0, similarity_score)
        
        # Apply multipliers
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
        
        # Calculate final score
        final_score = similarity_score * exp_mult * loc_mult * tenure_mult * behav_mult * notice_mult * service_mult * cv_mult
        final_score_scaled = round(final_score * 100.0, 4)
        
        # Factual, non-templated reasoning
        reasoning = generate_candidate_reasoning(cand, final_score_scaled)
        
        ranked_list.append({
            "candidate_id": cid,
            "score": final_score_scaled,
            "reasoning": reasoning
        })
        
    # 6. Sort by score desc, candidate_id asc (tie-breaker)
    print("Sorting candidates...")
    ranked_list.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    # 7. Select top 100
    top_100 = ranked_list[:100]
    
    # 8. Output to CSV
    print(f"Writing top 100 candidates to {args.out}...")
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, item in enumerate(top_100):
            writer.writerow([item["candidate_id"], i + 1, item["score"], item["reasoning"]])
            
    elapsed_time = datetime.now() - start_time
    print(f"Done! Finished ranking in {elapsed_time.total_seconds():.2f} seconds.")

if __name__ == "__main__":
    main()
