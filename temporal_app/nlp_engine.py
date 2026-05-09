import re
from transformers import pipeline

model_name = "davlan/distilbert-base-multilingual-cased-ner-hrl"
nlp = pipeline("ner", model=model_name, aggregation_strategy="simple")


# ─────────────────────────────────────────────
# LAYER 1 — Regex patterns
# ─────────────────────────────────────────────

MONTH_NAMES = (
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December'
    r'|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec'
    r'|जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर'
    r'|জানুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|মে|জুন|জুলাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর)'
)

ENG_NUMBERS = (
    r'(?:one|two|three|four|five|six|seven|eight|nine|ten'
    r'|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty)'
)

DATE_PATTERNS = [

    # ── Absolute dates (English) ────────────────────────────────────

    # 29 June 2023 / 29th June / June 29, 2023
    (r'\b\d{1,2}(?:st|nd|rd|th)?\s+' + MONTH_NAMES + r'(?:\s+\d{2,4})?\b', 'DATE'),
    (MONTH_NAMES + r'\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{2,4})?\b', 'DATE'),

    # DD/MM/YYYY, MM-DD-YYYY, YYYY.MM.DD
    (r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b', 'DATE'),

    # YYYY-MM-DD ISO
    (r'\b\d{4}-\d{2}-\d{2}\b', 'DATE'),

    # Year alone
    (r'\b(?:19|20)\d{2}\b', 'DATE'),

    # Month + Year
    (MONTH_NAMES + r'\s+(?:19|20)\d{2}\b', 'DATE'),

    # Quarter
    (r'\bQ[1-4]\s+(?:of\s+)?(?:19|20)\d{2}\b', 'DATE'),

    # Mid/early/late + month
    (r'\b(?:mid|early|late)[- ]' + MONTH_NAMES + r'\b', 'DATE'),

    # "first/second/third week of January"
    (r'\b(?:first|second|third|fourth|last)\s+week\s+of\s+' + MONTH_NAMES + r'\b', 'DATE'),

    # "third Friday of March", "second Monday of April"
    (r'\b(?:first|second|third|fourth|last)\s+'
     r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)'
     r'\s+of\s+' + MONTH_NAMES + r'\b', 'DATE'),

    # ── Relative dates (English) ────────────────────────────────────

    # "3 days ago", "two weeks ago"
    (r'\b(?:a\s+)?(?:\d+|' + ENG_NUMBERS + r')\s+(?:day|week|month|year)s?\s+ago\b', 'DATE'),

    # "in 3 days", "in two weeks"
    (r'\bin\s+(?:a\s+)?(?:\d+|' + ENG_NUMBERS + r')\s+(?:day|week|month|year)s?\b', 'DATE'),

    # "3 days later", "two weeks from now"
    (r'\b(?:\d+|' + ENG_NUMBERS + r')\s+(?:day|week|month|year)s?\s+(?:later|from now)\b', 'DATE'),

    # "next/last/this/coming Friday", "next month", "last year"
    (r'\b(?:next|last|this|coming|previous|past|upcoming)\s+'
     r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday'
     r'|week|month|year|weekend|quarter|semester|decade|fortnight)\b', 'DATE'),

    # "the day after tomorrow", "the day before yesterday"
    (r'\bthe\s+day\s+(?:after\s+tomorrow|before\s+yesterday)\b', 'DATE'),

    # "every Monday", "every week", "every year"
    (r'\bevery\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday'
     r'|day|week|month|year|morning|evening|night|afternoon)\b', 'DATE'),

    # yesterday / today / tomorrow
    (r'\b(?:yesterday|today|tomorrow)\b', 'DATE'),

    # "end of this month", "start of next year"
    (r'\b(?:end|start|beginning|middle)\s+of\s+(?:this|next|last)\s+'
     r'(?:week|month|year|quarter|semester)\b', 'DATE'),

    # "last hour", "past few days"
    (r'\b(?:last|past)\s+(?:few\s+)?(?:hour|day|week|month|year)s?\b', 'DATE'),

    # ── Hindi date patterns ─────────────────────────────────────────

    # "3 दिन पहले", "दो हफ्ते पहले", "5 महीने पहले"
    (r'\b(?:\d+|एक|दो|तीन|चार|पाँच|पांच|छह|सात|आठ|नौ|दस)\s+'
     r'(?:दिन|हफ्ते|हफ़्ते|महीने|साल|वर्ष|घंटे|मिनट)\s+(?:पहले|बाद|से)\b', 'DATE'),

    # "अगले/पिछले X दिन/हफ्ते/महीने"
    (r'(?:अगले|पिछले|इस|आने वाले)\s+(?:\d+\s+)?'
     r'(?:दिन|हफ्ते|हफ़्ते|महीने|साल|वर्ष|सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार|दशक)\b', 'DATE'),

    # Hindi numeric time — "9 बजे", "10 बजे", "साढ़े 3 बजे"
    (r'\b(?:साढ़े\s+)?(?:\d+|एक|दो|तीन|चार|पाँच|पांच|छह|सात|आठ|नौ|दस|ग्यारह|बारह)\s+बजे\b', 'TIME'),

    # "15 अगस्त", "26 जनवरी" — Hindi date with Hindi month
    (r'\b\d{1,2}\s+(?:जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\b', 'DATE'),

    # ── Bengali date patterns ───────────────────────────────────────

    # "তিন দিন আগে", "দুই সপ্তাহ আগে", "৩ মাস পরে"
    (r'(?:[\u09E6-\u09EF\d]+|এক|দুই|তিন|চার|পাঁচ|ছয়|সাত|আট|নয়|দশ)\s+'
     r'(?:দিন|সপ্তাহ|মাস|বছর|ঘণ্টা|মিনিট)\s+(?:আগে|পরে|ধরে|থেকে)\b', 'DATE'),

    # "আগামী/গত X দিন/সপ্তাহ"
    (r'(?:আগামী|গত|এই|পরশু|আগামী পরশু)\s+(?:[\u09E6-\u09EF\d]+\s+)?'
     r'(?:দিন|সপ্তাহ|মাস|বছর|সোমবার|মঙ্গলবার|বুধবার|বৃহস্পতিবার|শুক্রবার|শনিবার|রবিবার)\b', 'DATE'),

    # Bengali numeric time — "৬টায়", "১০টায়", "6টায়"
    (r'(?:[\u09E6-\u09EF\d]+)টায়\b', 'TIME'),

    # Bengali "প্রতিদিন", "প্রতি সপ্তাহ"
    (r'প্রতি(?:দিন|সপ্তাহ|মাস|বছর|(?:\s+(?:সোমবার|মঙ্গলবার|বুধবার|বৃহস্পতিবার|শুক্রবার|শনিবার|রবিবার)))\b', 'DATE'),

    # ── Time expressions (English) ──────────────────────────────────

    # 10:30 AM, 23:59, 9:00 p.m.
    (r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)?\b', 'TIME'),

    # "at 9 AM", "by 6 PM", "around 3 PM"
    (r'\b(?:at|by|around|before|after|until|till)\s+\d{1,2}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)\b', 'TIME'),

    # "morning", "afternoon", "evening", "midnight", "noon"
    (r'\b(?:early\s+)?(?:morning|afternoon|evening|night|midnight|noon|dawn|dusk|sunrise|sunset|tonight|tonite)\b', 'TIME'),

    # "3 hours ago", "in 30 minutes", "2 seconds later"
    (r'\b(?:\d+|a|an|' + ENG_NUMBERS + r')\s+(?:hour|minute|second)s?\s+(?:ago|later|from now)\b', 'TIME'),

    # "under four hours", "within two hours"
    (r'\b(?:under|within|less than|more than)\s+(?:\d+|' + ENG_NUMBERS + r')\s+(?:hour|minute|day|week)s?\b', 'TIME'),

    # ── Duration expressions (English) ─────────────────────────────

    # "for 3 days", "for two weeks", "for a month"
    (r'\bfor\s+(?:a\s+)?(?:\d+|' + ENG_NUMBERS + r')\s+'
     r'(?:day|week|month|year|hour|minute|second)s?\b', 'DURATION'),

    # "since 2020", "since last year", "since Monday", "since yesterday"
    (r'\bsince\s+(?:(?:19|20)\d{2}|(?:last|this)\s+\w+|yesterday|'
     r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))\b', 'DURATION'),

    # "over the past X days/years"
    (r'\bover\s+(?:the\s+)?(?:past|last)\s+(?:\d+|' + ENG_NUMBERS + r')\s+'
     r'(?:day|week|month|year)s?\b', 'DURATION'),

    # "throughout the year", "all week", "all month"
    (r'\b(?:throughout|all)\s+(?:the\s+)?(?:day|week|month|year|semester|decade)\b', 'DURATION'),
]


# ─────────────────────────────────────────────
# LAYER 2 — Multilingual keyword gazetteer
# ─────────────────────────────────────────────

TEMPORAL_KEYWORDS = {
    'DATE': [
        # ── Hindi days ──
        'सोमवार', 'मंगलवार', 'बुधवार', 'गुरुवार', 'शुक्रवार', 'शनिवार', 'रविवार',

        # ── Hindi relative — simple ──
        'कल', 'आज', 'परसों', 'नरसों',

        # ── Hindi relative — compound ──
        'अगले हफ्ते', 'पिछले हफ्ते', 'इस हफ्ते',
        'अगले महीने', 'पिछले महीने', 'इस महीने',
        'अगले साल', 'पिछले साल', 'इस साल',
        'अगले सोमवार', 'अगले मंगलवार', 'अगले बुधवार',
        'अगले गुरुवार', 'अगले शुक्रवार', 'अगले शनिवार', 'अगले रविवार',
        'पिछले सोमवार', 'पिछले मंगलवार', 'पिछले बुधवार',
        'आज सुबह', 'कल सुबह', 'कल शाम', 'कल रात',
        'आने वाले दिनों में', 'अगले दशक', 'पिछले दशक',

        # ── Hindi duration phrases ──
        'दो हफ्ते बाद', 'दो हफ्ते पहले',
        'दो महीने पहले', 'तीन महीने पहले',
        'पिछले तीन सालों से', 'पिछले दो सालों से',
        'कुछ दिनों में', 'कुछ दिन पहले',
        'थोड़े दिनों में', 'कुछ घंटों में',

        # ── Bengali days ──
        'সোমবার', 'মঙ্গলবার', 'বুধবার', 'বৃহস্পতিবার', 'শুক্রবার', 'শনিবার', 'রবিবার',

        # ── Bengali relative — simple ──
        'আজকে', 'আগামীকাল', 'গতকাল', 'পরশু', 'গতপরশু',

        # ── Bengali relative — compound ──
        'আগামী সপ্তাহ', 'গত সপ্তাহ', 'এই সপ্তাহ',
        'আগামী মাসে', 'গত মাসে', 'এই মাসে',
        'আগামী বছর', 'গত বছর', 'এই বছর',
        'আগামী পরশু', 'গত সোমবার', 'আগামী শুক্রবার',
        'আগামী সোমবার', 'আগামী মঙ্গলবার', 'আগামী বুধবার',
        'আগামী বৃহস্পতিবার', 'আগামী শনিবার', 'আগামী রবিবার',

        # ── Bengali duration phrases ──
        'দুই সপ্তাহ আগে', 'তিন সপ্তাহ আগে',
        'দুই মাস আগে', 'তিন মাস আগে',
        'গত দুই বছর ধরে', 'গত তিন বছর ধরে',
        'আজ থেকে তিন মাস পরে', 'আজ থেকে দুই সপ্তাহ পরে',
        'তিন দিন পরে', 'দুই দিন পরে',

        # ── English days ──
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
        'weekend', 'weekday', 'weekends', 'weekdays',
    ],

    'TIME': [
        # ── Hindi time of day ──
        'सुबह', 'दोपहर', 'शाम', 'रात', 'मध्यरात्रि', 'आधी रात', 'तड़के',

        # ── Bengali time of day ──
        'সকাল', 'দুপুর', 'বিকেল', 'রাত', 'মধ্যরাত', 'সন্ধ্যা', 'ভোর',
        'সন্ধ্যায়',

        # ── Bengali everyday time expressions ──
        'প্রতিদিন', 'প্রতিদিন সকালে', 'প্রতিদিন রাতে',
        'প্রতি সকালে', 'প্রতি রাতে',
    ],

    'DURATION': [
        # ── Hindi duration ──
        'कुछ दिनों से', 'लंबे समय से', 'थोड़े समय के लिए',
        'काफी समय से', 'बहुत दिनों से', 'सालों से',
        'घंटों से', 'महीनों से',

        # ── Bengali duration ──
        'কিছুদিন ধরে', 'দীর্ঘদিন ধরে', 'অনেকদিন ধরে',
        'কয়েক বছর ধরে', 'কয়েক মাস ধরে', 'কয়েক সপ্তাহ ধরে',
        'তিন ঘণ্টা আগে', 'দুই ঘণ্টা আগে', 'এক ঘণ্টা আগে',
        'কিছুক্ষণ আগে', 'অনেকক্ষণ আগে',
    ],
}


# ─────────────────────────────────────────────
# Helper — smarter dedup
# ─────────────────────────────────────────────

def _already_found(results, word):
    """Return True if word is already covered by an existing result."""
    word_lower = word.lower().strip()
    for res in results:
        existing = res['word'].lower().strip()
        if existing == word_lower:
            return True
        if word_lower in existing:
            return True
    return False


# ─────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────

def get_temporal_entities(text):
    """
    Extract temporal entities from text using three layers:
      1. Transformer NER  — multilingual DistilBERT
      2. Regex patterns   — 30+ patterns, English + Hindi + Bengali
      3. Keyword gazetteer — expanded multilingual vocabulary
    """

    # Layer 1 — transformer NER
    results = nlp(text)

    # Layer 2 — regex (longer/more specific patterns first)
    for pattern, label in DATE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = ' '.join(m for m in match if m).strip()
            else:
                match = match.strip()
            if match and not _already_found(results, match):
                results.append({
                    'entity_group': label,
                    'word': match,
                    'score': 1.0,
                })

    # Layer 3 — gazetteer (longer phrases first to avoid partial overlaps)
    for label, keywords in TEMPORAL_KEYWORDS.items():
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        for kw in sorted_keywords:
            if kw.lower() in text.lower() and not _already_found(results, kw):
                results.append({
                    'entity_group': label,
                    'word': kw,
                    'score': 1.0,
                })

    return results