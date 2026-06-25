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

def is_honeypot(candidate) -> tuple:
    """
    Checks if a candidate profile has anomalies that indicate it is a honeypot.
    Returns (is_anomaly, reason)
    """
    cid = candidate.get('candidate_id', '')
    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    
    # 1. Skill Fraud: Expert/Advanced with 0 months
    for s in skills:
        prof = str(s.get('proficiency', '')).lower()
        dur = s.get('duration_months')
        if prof in ['expert', 'advanced'] and dur == 0:
            return True, f"Skill fraud: {s.get('name')} is {prof} but 0 months"
            
    # 2. Career duration vs start/end dates
    for i, job in enumerate(career):
        start = parse_date(job.get('start_date'))
        end = parse_date(job.get('end_date')) if job.get('end_date') else datetime(2026, 6, 25)
        duration = job.get('duration_months', 0)
        
        span_months = (end.year - start.year) * 12 + (end.month - start.month)
        if abs(span_months - duration) > 12:
            return True, f"Career[{i}] duration mismatch: dates span {span_months} months, but duration_months={duration}"
            
    # 3. YoE mismatch vs Career History sum
    total_career_months = sum(j.get('duration_months', 0) for j in career)
    yoe = profile.get('years_of_experience', 0)
    if abs((total_career_months / 12.0) - yoe) > 3.0:
        return True, f"YoE mismatch: stated={yoe}, sum of history={total_career_months / 12.0:.2f} years"
        
    return False, ""
