import os
import json
import time
import gzip
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from utils.honeypots import is_honeypot

candidates_path = "./India_runs_data_and_ai_challenge/candidates.jsonl"
output_cache_path = "./precomputed_data.pkl.gz"
local_model_path = "./models/all-MiniLM-L6-v2"

def main():
    start_time = time.time()
    
    # 1. Load and save the model locally for complete offline access
    print("Loading all-MiniLM-L6-v2 from HF Hub...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Saving model to local path: {local_model_path}")
    os.makedirs(local_model_path, exist_ok=True)
    model.save(local_model_path)
    
    # Reload model from local path to verify offline integrity
    print("Reloading model from local directory to verify offline loading...")
    model = SentenceTransformer(local_model_path)
    print("Local model verified successfully!")
    
    # Set CPU threads to 4 for optimal performance
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        torch.set_num_threads(4)
    print(f"Using device for embedding generation: {device}")
    model.to(device)
    
    candidate_ids = []
    texts_to_embed = []
    honeypots_dict = {}
    
    print("Reading candidates from JSONL...")
    candidate_count = 0
    
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            cand = json.loads(line)
            cid = cand['candidate_id']
            profile = cand.get('profile', {})
            career = cand.get('career_history', [])
            skills = cand.get('skills', [])
            
            # Check if candidate is a honeypot
            is_hp, reason = is_honeypot(cand)
            honeypots_dict[cid] = (is_hp, reason)
            
            # Create text representation
            current_title = profile.get('current_title', '')
            headline = profile.get('headline', '')
            
            text_rep = f"Title: {current_title}. Headline: {headline}. "
            
            # Select top 15 skills to keep sequence length short and highly relevant
            top_skills = [s.get('name', '') for s in skills if s.get('name')]
            skills_str = ", ".join(top_skills[:15])
            if skills_str:
                text_rep += f"Skills: {skills_str}. "
                
            roles = ", ".join([job.get('title', '') for job in career if job.get('title')])
            if roles:
                text_rep += f"Roles: {roles}."
                
            # Truncate to save memory
            text_rep = text_rep[:400]
            
            candidate_ids.append(cid)
            texts_to_embed.append(text_rep)
            
            candidate_count += 1
            if candidate_count % 20000 == 0:
                print(f"Read {candidate_count} profiles...")
                
    print(f"Total candidates read: {candidate_count}")
    print("Generating dense vector embeddings in batches...")
    
    # Generate embeddings
    batch_size = 512 if device == "cuda" else 128
    embeddings = model.encode(
        texts_to_embed, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    
    print(f"Embeddings generated with shape: {embeddings.shape}")
    
    # Save cache file
    print(f"Saving precomputed data cache to {output_cache_path}...")
    data_to_cache = {
        "candidate_ids": candidate_ids,
        "embeddings": embeddings.astype(np.float32),  # Ensure float32 for space efficiency
        "honeypots": honeypots_dict
    }
    
    with gzip.open(output_cache_path, "wb") as f:
        pickle.dump(data_to_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    elapsed = time.time() - start_time
    print(f"Precomputation complete! Total time: {elapsed/60:.2f} minutes.")
    print(f"Cache file size: {os.path.getsize(output_cache_path) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
