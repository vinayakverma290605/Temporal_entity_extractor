import re
from transformers import pipeline

# This distilled multilingual model is stable on Mac and supports 100+ languages
model_name = "davlan/distilbert-base-multilingual-cased-ner-hrl"
nlp = pipeline("ner", model=model_name, aggregation_strategy="simple")

def get_temporal_entities(text):
    # 1. AI detection (Finds Bharat, Delhi, Vinayak, etc.)
    results = nlp(text)
    
    # 2. Regex Pattern for Dates (Catching "29 June", "15 Aug", etc.)
    # This looks for: 1 or 2 digits + space + Month Name
    date_pattern = r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Oct|Nov|Dec))'
    
    found_dates = re.findall(date_pattern, text, re.IGNORECASE)
    for date_word in found_dates:
        # Avoid duplicates if the AI happens to catch it too
        if not any(res['word'].lower() == date_word.lower() for res in results):
            results.append({
                'entity_group': 'DATE',
                'word': date_word,
                'score': 1.0
            })
    
    # 3. Hindi/Bengali/English Keyword Gazetteer
    temporal_keywords = {
        'DATE': [
            'मंगलवार', 'सोमवार', 'बुधवार', 'गुरुवार', 'शनिवार', 'रविवार', 
            'মঙ্গলবার', 'সোমবার', 'বুধবার', 'বৃহস্পতিবার', 'শুক্রবার', 'শনিবার', 'রবিবার',
            'August', 'अगस्त', 'আগস্ট', 'कल', 'आज', 'আগামীকাল', 'আজকে',
            'Friday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday', 'Sunday'
        ]
    }
    
    # Check for keywords and add them if not already detected
    for category, words in temporal_keywords.items():
        for word in words:
            if word.lower() in text.lower():
                if not any(res['word'].lower() == word.lower() for res in results):
                    results.append({
                        'entity_group': category,
                        'word': word,
                        'score': 1.0
                    })
            
    return results