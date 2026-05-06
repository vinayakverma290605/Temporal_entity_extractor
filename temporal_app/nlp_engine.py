import re
from transformers import pipeline

# Same model as before — already downloaded, no new install needed
model_name = "davlan/distilbert-base-multilingual-cased-ner-hrl"
nlp = pipeline("ner", model=model_name, aggregation_strategy="simple")


# ─────────────────────────────────────────────
# LAYER 1 — Regex patterns (greatly expanded)
# ─────────────────────────────────────────────

MONTH_NAMES = (
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December'
    r'|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec'
    r'|जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर'
    r'|জানুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|মে|জুন|জুলাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর)'
)

DATE_PATTERNS = [

    # ── Absolute dates ──────────────────────────────────────────────

    # 29 June 2023 / 29th June 2023 / June 29, 2023
    (r'\b\d{1,2}(?:st|nd|rd|th)?\s+' + MONTH_NAMES + r'(?:\s+\d{2,4})?\b', 'DATE'),
    (MONTH_NAMES + r'\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{2,4})?\b', 'DATE'),

    # DD/MM/YYYY, MM-DD-YYYY, YYYY.MM.DD etc.
    (r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b', 'DATE'),

    # YYYY-MM-DD (ISO)
    (r'\b\d{4}-\d{2}-\d{2}\b', 'DATE'),

    # Year alone: 2023, 1998
    (r'\b(?:19|20)\d{2}\b', 'DATE'),

    # Month + Year: "June 2023", "अगस्त 2022"
    (MONTH_NAMES + r'\s+(?:19|20)\d{2}\b', 'DATE'),

    # Quarter: Q1 2024, Q3 of 2023
    (r'\bQ[1-4]\s+(?:of\s+)?(?:19|20)\d{2}\b', 'DATE'),

    # Mid/early/late + month: "mid-January", "late August"
    (r'\b(?:mid|early|late)[- ]' + MONTH_NAMES + r'\b', 'DATE'),

    # ── Relative date expressions ───────────────────────────────────

    # "3 days ago", "2 weeks ago", "a month ago"
    (r'\b(?:a\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+'
     r'(?:day|week|month|year)s?\s+ago\b', 'DATE'),

    # "in 3 days", "in two weeks", "in a month"
    (r'\bin\s+(?:a\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+'
     r'(?:day|week|month|year)s?\b', 'DATE'),

    # "next Friday", "last Monday", "this weekend", "next month", "last year"
    (r'\b(?:next|last|this|coming|previous|past)\s+'
     r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday'
     r'|week|month|year|weekend|quarter|semester|decade)\b', 'DATE'),

    # "the day after tomorrow", "the day before yesterday"
    (r'\bthe\s+day\s+(?:after\s+tomorrow|before\s+yesterday)\b', 'DATE'),

    # yesterday / today / tomorrow
    (r'\b(?:yesterday|today|tomorrow)\b', 'DATE'),

    # ── Time expressions ────────────────────────────────────────────

    # 10:30 AM, 23:59, 9:00 p.m.
    (r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)?\b', 'TIME'),

    # "at 9 AM", "by 6 PM"
    (r'\b(?:at|by|around|before|after)\s+\d{1,2}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)\b', 'TIME'),

    # "morning", "afternoon", "evening", "midnight", "noon"
    (r'\b(?:early\s+)?(?:morning|afternoon|evening|night|midnight|noon|dawn|dusk|sunrise|sunset)\b', 'TIME'),

    # "3 hours ago", "in 30 minutes"
    (r'\b(?:\d+|a|an)\s+(?:hour|minute|second)s?\s+(?:ago|later|from now)\b', 'TIME'),

    # ── Duration expressions ────────────────────────────────────────

    # "for 3 days", "for two weeks", "for a month"
    (r'\bfor\s+(?:a\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+'
     r'(?:day|week|month|year|hour|minute)s?\b', 'DURATION'),

    # "since 2020", "since last year", "since Monday"
    (r'\bsince\s+(?:(?:19|20)\d{2}|last\s+\w+|(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))\b', 'DURATION'),

    # "from X to Y" (rough duration range)
    (r'\bfrom\s+\w+\s+to\s+\w+\b', 'DURATION'),
]

# ─────────────────────────────────────────────
# LAYER 2 — Multilingual keyword gazetteer
# ─────────────────────────────────────────────

TEMPORAL_KEYWORDS = {
    'DATE': [
        # Hindi days
        'सोमवार', 'मंगलवार', 'बुधवार', 'गुरुवार', 'शुक्रवार', 'शनिवार', 'रविवार',
        # Hindi relative
        'कल', 'आज', 'परसों', 'अगले हफ्ते', 'पिछले हफ्ते', 'इस महीने',
        'अगले महीने', 'पिछले महीने', 'इस साल', 'अगले साल', 'पिछले साल',
        'आज सुबह', 'कल शाम', '3 दिन पहले', 'अगले सोमवार',
        # Bengali days
        'সোমবার', 'মঙ্গলবার', 'বুধবার', 'বৃহস্পতিবার', 'শুক্রবার', 'শনিবার', 'রবিবার',
        # Bengali relative
        'আজকে', 'আগামীকাল', 'গতকাল', 'আগামী সপ্তাহ', 'গত সপ্তাহ',
        'এই মাসে', 'আগামী মাসে', 'গত মাসে',
        # English days (catches standalone)
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
        'weekend', 'weekday',
    ],
    'TIME': [
        # Hindi
        'सुबह', 'दोपहर', 'शाम', 'रात', 'मध्यरात्रि',
        # Bengali
        'সকাল', 'দুপুর', 'বিকেল', 'রাত', 'মধ্যরাত',
    ],
    'DURATION': [
        # Hindi
        'कुछ दिनों से', 'लंबे समय से', 'थोड़े समय के लिए',
        # Bengali
        'কিছুদিন ধরে', 'দীর্ঘদিন ধরে',
    ],
}


# ─────────────────────────────────────────────
# Helper — dedup check
# ─────────────────────────────────────────────

def _already_found(results, word):
    """Return True if this word/phrase is already in results (case-insensitive)."""
    word_lower = word.lower().strip()
    for res in results:
        if res['word'].lower().strip() == word_lower:
            return True
        # Also skip if it's a substring of something already found
        if word_lower in res['word'].lower():
            return True
    return False


# ─────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────

def get_temporal_entities(text):
    """
    Extract temporal entities from text using three layers:
      1. Transformer NER (catches named entities as context)
      2. Regex patterns (dates, times, durations — English + Hindi + Bengali)
      3. Keyword gazetteer (multilingual day/time words)
    """

    # Layer 1 — transformer NER (keep for context, e.g. event names with dates)
    results = nlp(text)

    # Layer 2 — regex
    for pattern, label in DATE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            match = match.strip()
            if match and not _already_found(results, match):
                results.append({
                    'entity_group': label,
                    'word': match,
                    'score': 1.0,
                })

    # Layer 3 — gazetteer
    for label, keywords in TEMPORAL_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text.lower() and not _already_found(results, kw):
                results.append({
                    'entity_group': label,
                    'word': kw,
                    'score': 1.0,
                })

    return results