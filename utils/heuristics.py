import re
from datetime import datetime

def parse_date(date_str, default_date="2026-06-25"):
    if not date_str:
        return datetime.strptime(default_date, "%Y-%m-%d")
    date_str = str(date_str).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return datetime.strptime(default_date, "%Y-%m-%d")

def check_service_only(career_history) -> bool:
    """
    Returns True if the candidate has ONLY worked at service/consulting companies.
    """
    if not career_history:
        return False
    service_keywords = {
        'tcs', 'tata consultancy', 'tata consultancy services', 'infosys', 'wipro', 
        'accenture', 'cognizant', 'capgemini', 'hcl', 'tech mahindra', 'mindtree', 
        'l&t', 'larsen & toubro', 'dxc', 'deloitte', 'ey', 'pwc', 'kpmg', 'ibm', 
        'cognizant technology solutions', 'genpact', 'infosys bpm', 'wipro technologies'
    }
    for job in career_history:
        comp = str(job.get('company', '')).strip().lower()
        # Clean company name
        comp_clean = re.sub(r'\b(pvt|ltd|inc|corp|co|limited|private|services|technologies|solutions)\b', '', comp).strip()
        
        is_service = False
        for kw in service_keywords:
            if kw in comp or comp in kw or (len(comp_clean) > 2 and comp_clean in kw):
                is_service = True
                break
        if not is_service:
            # Found a product or non-service company
            return False
    return True

def calculate_tenure_multiplier(career_history) -> float:
    """
    Returns tenure multiplier (penalizes job hoppers switching every <1.5 years on average).
    """
    if not career_history:
        return 1.0
    total_months = 0
    num_jobs = 0
    for job in career_history:
        dur = job.get('duration_months', 0)
        total_months += dur
        num_jobs += 1
    if num_jobs <= 1:
        return 1.0
    avg_tenure_years = (total_months / num_jobs) / 12.0
    if avg_tenure_years < 1.5:
        # Linear scale down from 1.5 years (1.0 multiplier) to 0.5 years (0.5 multiplier)
        return max(0.5, avg_tenure_years / 1.5)
    return 1.0

def calculate_experience_multiplier(years_of_experience) -> float:
    """
    Calculates multiplier based on experience fit (target 5-9 years).
    """
    yoe = years_of_experience
    if 5.0 <= yoe <= 9.0:
        return 1.0
    elif 4.0 <= yoe < 5.0:
        return 0.8
    elif 9.0 < yoe <= 12.0:
        return 0.9
    elif 3.0 <= yoe < 4.0:
        return 0.5
    elif 12.0 < yoe <= 15.0:
        return 0.6
    else:
        return 0.2

def calculate_location_multiplier(location, country, willing_to_relocate) -> float:
    """
    Calculates multiplier based on location fit (target Pune/Noida).
    """
    loc = str(location).strip().lower()
    ctry = str(country).strip().lower()
    
    # Local check (Noida, Pune, Delhi NCR)
    if any(city in loc for city in ['pune', 'noida', 'delhi', 'gurgaon', 'ncr', 'ghaziabad', 'faridabad']):
        return 1.0
        
    # Tier-1 Indian cities
    tier1_cities = {'hyderabad', 'mumbai', 'bangalore', 'bengaluru', 'chennai', 'kolkata'}
    is_tier1 = any(city in loc for city in tier1_cities)
    
    # Check if candidate is in India
    is_india = (ctry == 'india' or is_tier1)
    
    if is_india:
        if is_tier1:
            return 0.8 if willing_to_relocate else 0.3
        else:
            return 0.5 if willing_to_relocate else 0.1
    else:
        # Outside India
        return 0.2 if willing_to_relocate else 0.0

def calculate_behavioral_multiplier(signals) -> float:
    """
    Calculates multiplier based on platform active status and engagement.
    """
    if not signals:
        return 0.5
        
    multiplier = 1.0
    
    # 1. Recruiter Response Rate
    response_rate = signals.get('recruiter_response_rate', 1.0)
    if response_rate < 0.15:
        multiplier *= 0.6
    elif response_rate < 0.40:
        multiplier *= 0.85
    elif response_rate >= 0.80:
        multiplier *= 1.1
        
    # 2. Last active date (target: active in last 6 months)
    last_active_str = signals.get('last_active_date', '')
    if last_active_str:
        last_active = parse_date(last_active_str)
        ref_date = datetime(2026, 6, 25)
        days_inactive = (ref_date - last_active).days
        if days_inactive > 180:
            multiplier *= 0.5
        elif days_inactive > 90:
            multiplier *= 0.8
        elif days_inactive <= 30:
            multiplier *= 1.05
            
    # 3. Open to work flag
    otw = signals.get('open_to_work_flag', False)
    if otw:
        multiplier *= 1.1
    else:
        multiplier *= 0.9
        
    # 4. Profile completeness
    completeness = signals.get('profile_completeness_score', 100.0)
    if completeness < 50.0:
        multiplier *= 0.7
    elif completeness < 75.0:
        multiplier *= 0.9
        
    # 5. Interview completion rate
    icr = signals.get('interview_completion_rate', 1.0)
    if icr < 0.50:
        multiplier *= 0.7
        
    return max(0.1, multiplier)

def check_cv_speech_bias(skills) -> bool:
    """
    Returns True if the candidate only has CV/speech/robotics skills and lacks NLP/IR.
    """
    cv_speech_skills = {
        'computer vision', 'image classification', 'object detection', 'segmentation', 
        'speech recognition', 'tts', 'text to speech', 'speech to text', 'audio processing', 
        'robotics', 'ros', 'slam', 'opencv', 'pytorch cv', 'tensorflow cv'
    }
    nlp_ir_skills = {
        'nlp', 'natural language processing', 'embeddings', 'retrieval', 'search', 
        'vector search', 'information retrieval', 'ranking', 'bm25', 'elasticsearch', 
        'opensearch', 'hybrid search', 'milvus', 'qdrant', 'pinecone', 'weaviate', 
        'llm', 'llms', 'fine-tuning llms', 'transformer', 'transformers', 'bert', 
        'gpt', 'langchain', 'llama-index', 'vector database', 'vector databases'
    }
    
    has_cv_speech = False
    has_nlp_ir = False
    
    for s in skills:
        name = str(s.get('name', '')).strip().lower()
        if name in cv_speech_skills or any(k in name for k in cv_speech_skills):
            has_cv_speech = True
        if name in nlp_ir_skills or any(k in name for k in nlp_ir_skills):
            has_nlp_ir = True
            
    if has_cv_speech and not has_nlp_ir:
        return True
    return False
