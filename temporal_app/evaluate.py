import os
import sys

# This lets evaluate.py find nlp_engine.py even when run from terminal
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nlp_engine import get_temporal_entities

# ── Load files ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "test_sentences.txt"), encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

with open(os.path.join(BASE, "answer_key.txt"), encoding="utf-8") as f:
    answer_key = [line.strip() for line in f if line.strip()]

assert len(sentences) == len(answer_key), "Mismatch: sentences and answer key must have same number of lines!"

# ── Helpers ─────────────────────────────────────────────────
def normalize(text):
    return text.lower().strip()

def is_match(expected, detected_list):
    e = normalize(expected)
    return any(e in normalize(d) or normalize(d) in e for d in detected_list)

# ── Per language tracking ────────────────────────────────────
languages = {
    "English": {"expected": 0, "detected": 0, "correct": 0},
    "Hindi":   {"expected": 0, "detected": 0, "correct": 0},
    "Bengali": {"expected": 0, "detected": 0, "correct": 0},
}

def detect_language(text):
    hindi_chars   = set("कखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहािीुूेैोौंःअआइईउऊ")
    bengali_chars = set("অআইঈউঊএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ")
    if any(c in hindi_chars for c in text):
        return "Hindi"
    if any(c in bengali_chars for c in text):
        return "Bengali"
    return "English"

# ── Run evaluation ───────────────────────────────────────────
total_expected = 0
total_detected = 0
total_correct  = 0

print("\n" + "="*65)
print("   TEMPORAL ENTITY EXTRACTOR — EVALUATION REPORT")
print("="*65)

for i, (sentence, answers_raw) in enumerate(zip(sentences, answer_key)):
    expected_list = [a.strip() for a in answers_raw.split("|")]
    lang = detect_language(sentence)

    # Run your system
    raw_results = get_temporal_entities(sentence)
    detected_words = [
        r["word"] for r in raw_results
        if r["entity_group"] in ["DATE", "TIME", "DURATION"]
    ]

    # Match
    correct_list = [e for e in expected_list if is_match(e, detected_words)]
    missed_list  = [e for e in expected_list if not is_match(e, detected_words)]

    # Count
    total_expected += len(expected_list)
    total_detected += len(detected_words)
    total_correct  += len(correct_list)

    languages[lang]["expected"] += len(expected_list)
    languages[lang]["detected"] += len(detected_words)
    languages[lang]["correct"]  += len(correct_list)

    status = "✓" if len(correct_list) == len(expected_list) else "✗"

    print(f"\n[{i+1:02d}] {status}  [{lang}] {sentence}")
    print(f"       Expected : {expected_list}")
    print(f"       Detected : {detected_words}")
    if missed_list:
        print(f"       MISSED   : {missed_list}")

# ── Overall metrics ──────────────────────────────────────────
def metrics(correct, detected, expected):
    p = correct / detected  if detected  > 0 else 0
    r = correct / expected  if expected  > 0 else 0
    f = (2*p*r)/(p+r)       if (p+r)    > 0 else 0
    return p, r, f

precision, recall, f1 = metrics(total_correct, total_detected, total_expected)

print("\n" + "="*65)
print("   OVERALL RESULTS")
print("="*65)
print(f"  Total expected : {total_expected}")
print(f"  Total detected : {total_detected}")
print(f"  Correct        : {total_correct}")
print(f"  Precision      : {precision:.2%}")
print(f"  Recall         : {recall:.2%}")
print(f"  F1 Score       : {f1:.2%}")

print("\n" + "="*65)
print("   RESULTS BY LANGUAGE")
print("="*65)
for lang, counts in languages.items():
    p, r, f = metrics(counts["correct"], counts["detected"], counts["expected"])
    print(f"\n  {lang}")
    print(f"    Expected : {counts['expected']}  |  Detected : {counts['detected']}  |  Correct : {counts['correct']}")
    print(f"    Precision: {p:.2%}  |  Recall: {r:.2%}  |  F1: {f:.2%}")

print("\n" + "="*65)