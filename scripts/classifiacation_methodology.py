

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
from collections import Counter
import os

OUTPUT_DIR = '.'


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 – EXTRACT RAW A7 TEXT PER STUDENT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 1 – Extracting A7 raw text responses")
print("="*65)

df_raw = pd.read_excel('data_unical_2026-05-12_15-12.xlsx', header=None)
df = df_raw.iloc[2:].copy()
df.columns = df_raw.iloc[0].tolist()
df = df.reset_index(drop=True)
df = df[df['FINISHED'] == 1].copy().reset_index(drop=True)

TEXT_COLS = ['A701_01', 'A702_01', 'A703_01', 'A704_01', 'A705_01', 'A706_01']
TEXT_LABELS = {
    'A701_01': 'Q1: Innovate craft keeping identity',
    'A702_01': 'Q2: Elements of territory to valorise',
    'A703_01': 'Q3: Maximise perceived customer value',
    'A704_01': 'Q4: Sustainability as competitive advantage',
    'A705_01': 'Q5: Invest in 2030 Agenda pillar',
    'A706_01': 'Q6: Impact on local economy & tourism',
}

# Concatenate all 6 question texts per student into one blob for analysis
for col in TEXT_COLS:
    df[col] = df[col].fillna('').astype(str).str.strip()

df['ALL_TEXT'] = df[TEXT_COLS].apply(
    lambda row: ' ||| '.join([t for t in row if t and t.lower() != 'nan']),
    axis=1
)
df['TOTAL_CHARS'] = df['ALL_TEXT'].str.len()

print(f"\n  Students with completed surveys: {len(df)}")
print(f"  A7 questions per student: {len(TEXT_COLS)}")
print(f"  Mean total text length: {df['TOTAL_CHARS'].mean():.0f} chars")
print(f"  Range: {df['TOTAL_CHARS'].min()} – {df['TOTAL_CHARS'].max()} chars")


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 – DEFINE FINGERPRINTS: WHAT EACH GROUP LEAVES IN THE TEXT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 2 – Defining group fingerprints")
print("="*65)

FINGERPRINTS = {
    1: {
        'name': 'Knowledge Learning (Textbook)',
        'description': (
            'Students had access ONLY to the physical textbook. '
            'Expected signals: partial HE framework terms (picked up from reading), '
            'short or incomplete answers, no URLs, no AI-style citations, '
            'personal examples from local knowledge, may give up mid-survey.'
        ),
        'positive_signals': [
            'Uses some chapter vocabulary (Schumpeter, homo sentimentalis, etc.)',
            'Short concise answers — limited by what they could recall',
            'Personal, local examples (real Calabrian companies/places)',
            'Incomplete answers — single digits or "." when stuck',
            'No external URLs; no [n] citations; no "le fonti" language',
        ],
        'negative_signals': [
            'Will NOT have [1][2] citations or "Secondo le fonti"',
            'Will NOT have http:// URLs in answers',
            'Will NOT write full exhaustive HE framework coverage',
        ],
        'color': '#534AB7',
    },
    2: {
        'name': 'Information Extraction (NotebookLM)',
        'description': (
            'Students used NotebookLM with the textbook as source. '
            'NotebookLM has a distinctive output style: it cites its sources '
            'inline as [1],[2],[3], uses markdown formatting (### headers, ** bold), '
            'explicitly references "le fonti" / "i documenti", and produces '
            'exhaustive, structured summaries that stay 100% within the chapter.'
        ),
        'positive_signals': [
            'Inline citations [1] [2] [3] in the text',
            'References "le fonti", "secondo le fonti", "fonti fornite", "i documenti"',
            'Markdown ### headers and ** bold formatting',
            'Full exhaustive HE framework (Homo humus, Salus salutis, Homo humanitas)',
            'Very long responses — NotebookLM is thorough',
            'May explicitly say what the sources DO or DON\'T cover',
        ],
        'negative_signals': [
            'Will NOT have external URLs',
            'Will NOT go off-topic from Chapter 4 content',
        ],
        'color': '#0F6E56',
    },
    3: {
        'name': 'Knowledge Discovery (Web Search)',
        'description': (
            'Students used a search engine without AI assistance. '
            'They copy-paste or paraphrase web content. '
            'Most distinctive signal: explicit http:// URLs appear in answers. '
            'Content is generic business/marketing language from random websites, '
            'not from the HE framework. Some students submitted only URL lists.'
        ),
        'positive_signals': [
            'Explicit http:// URLs in answer text',
            'Copy-pasted blocks of web content (often in quotes)',
            'Generic digital marketing language (SEO, Google Ads, e-commerce hub)',
            'Off-topic answers (not from Chapter 4)',
            'Same text repeated across two questions (copy-paste error)',
            'Named external authors or sources (e.g. "Venegono")',
        ],
        'negative_signals': [
            'Will NOT use HE framework terminology',
            'Will NOT have [1][2] citations',
            'Will NOT reference "le fonti"',
        ],
        'color': '#993C1D',
    },
    4: {
        'name': 'Interaction (LLM)',
        'description': (
            'Students interacted with LLMs (ChatGPT/Claude/etc). '
            'LLM output is generative — it produces original, fluent text '
            'without citing a specific source. Signals: answers may be in English, '
            'use creative original concepts not in any source, follow a structured '
            'consulting/academic report format, or apply the topic to a specific '
            'company or analogy invented on the spot.'
        ),
        'positive_signals': [
            'Written in English (Italian students using English = LLM)',
            'Invented novel concepts not in textbook or web (e.g. "Craft Passport")',
            'Numbered consulting report format (2.1, 2.2, 3.1 sub-sections)',
            'Analysis centred on a specific named company (Osacuca, Feras)',
            'Fluent, balanced synthesis — no source, no HE jargon, no URLs',
            'Creative metaphors and original analogies',
        ],
        'negative_signals': [
            'Will NOT cite [1][2] or "le fonti"',
            'Will NOT include http:// URLs',
            'Will NOT use exact HE chapter terminology verbatim',
        ],
        'color': '#185FA5',
    },
}

for g, fp in FINGERPRINTS.items():
    print(f"\n  Group {g}: {fp['name']}")
    print(f"  {fp['description']}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 – COMPUTE 12 MEASURABLE SIGNALS PER STUDENT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 3 – Computing 12 measurable signals per student")
print("="*65)

# ── HE chapter vocabulary (Chapter 4 specific terms) ──────────────────────
HE_TERMS = [
    'imprenditore armonico', 'harmonic entrepreneur',
    'homo humus', 'homo humanitas', 'salus salutis',
    'tre punte', 'three-pronged', 'diamante',
    'eudaimonia', 'philautia', 'schumpeter',
    'doppio nesso', 'double nexus', 'co-evolv',
    'custode del creato', 'patermunus',
    'homo sentimentalis', 'capitale sociale',
    'rigenerazione', 'economia circolare',
    'imprenditorialità armonica', 'imprenditoria armonica',
]

# ── NotebookLM phrases ─────────────────────────────────────────────────────
FONTI_PHRASES = [
    'le fonti', 'delle fonti', 'secondo le fonti',
    'i documenti', 'nei documenti', 'fonti fornite',
    'sulla base delle fonti', 'secondo i documenti',
]

def compute_signals(text, col_texts):
    """Compute all 12 signals for a student's combined A7 text."""
    t = text.lower()

    # Signal 1: Total word count (all 6 questions combined)
    words = text.split()
    word_count = len(words)

    # Signal 2: Count of http:// URLs
    url_count = len(re.findall(r'https?://', text))

    # Signal 3: [1] [2] [3] style numbered citations
    cited_refs = len(re.findall(r'\[\d+\]', text))

    # Signal 4: "le fonti" / "i documenti" references
    fonti_refs = sum(t.count(phrase) for phrase in FONTI_PHRASES)

    # Signal 5: HE chapter vocabulary count
    he_keywords = sum(1 for term in HE_TERMS if term in t)

    # Signal 6: Markdown headers (### or ##)
    has_headers = int(bool(re.search(r'#{2,3}\s', text)))

    # Signal 7: Has bullet points or numbered lists
    has_bullets = int(bool(re.search(r'(\* |\• |\- |\d+\. )', text)))

    # Signal 8: Answers in English (>30% English words)
    english_words = {'the', 'and', 'for', 'that', 'with', 'this', 'from',
                     'have', 'are', 'not', 'you', 'can', 'will', 'but',
                     'craft', 'enterprise', 'sustainability', 'identity',
                     'innovation', 'territory', 'community', 'value',
                     'environment', 'social', 'local'}
    word_set = set(w.lower().strip('.,;:!?') for w in words)
    english_ratio = len(word_set & english_words) / max(len(word_set), 1)
    is_english = int(english_ratio > 0.08)

    # Signal 9: Number of questions answered with meaningful content (>30 chars)
    questions_answered = sum(
        1 for ct in col_texts
        if len(str(ct).strip()) > 30 and str(ct).strip().lower() != 'nan'
    )

    # Signal 10: Answers that are ONLY a number or single character (gave up)
    trivial_answers = sum(
        1 for ct in col_texts
        if re.match(r'^\s*\d\s*$', str(ct).strip()) or str(ct).strip() in ['.', '..', '']
    )

    # Signal 11: Copy-paste detected — same sentence block in multiple answers
    # Check if any 40-char substring from one answer appears in another
    copy_paste = 0
    texts_list = [str(ct).strip() for ct in col_texts if len(str(ct).strip()) > 40]
    for i in range(len(texts_list)):
        for j in range(i+1, len(texts_list)):
            snippet = texts_list[i][20:60]  # take a chunk from middle
            if snippet and snippet in texts_list[j]:
                copy_paste = 1
                break

    # Signal 12: Consulting report format (numbered sub-sections like 2.1, 3.2)
    report_format = int(bool(re.search(r'\d+\.\d+\s+\w', text)))

    return {
        'word_count'         : word_count,
        'url_count'          : url_count,
        'cited_refs'         : cited_refs,
        'fonti_refs'         : fonti_refs,
        'he_keywords'        : he_keywords,
        'has_headers'        : has_headers,
        'has_bullets'        : has_bullets,
        'is_english'         : is_english,
        'questions_answered' : questions_answered,
        'trivial_answers'    : trivial_answers,
        'copy_paste'         : copy_paste,
        'report_format'      : report_format,
    }

# Compute for all students
signals_list = []
for _, row in df.iterrows():
    col_texts = [row[col] for col in TEXT_COLS]
    sigs = compute_signals(row['ALL_TEXT'], col_texts)
    sigs['CASE'] = row['CASE']
    signals_list.append(sigs)

signals_df = pd.DataFrame(signals_list).set_index('CASE')

SIGNAL_DESCRIPTIONS = {
    'word_count'        : 'Total words in all 6 answers',
    'url_count'         : 'HTTP URLs found in text',
    'cited_refs'        : 'Inline [n] citations (NotebookLM style)',
    'fonti_refs'        : '"le fonti" / "i documenti" phrases',
    'he_keywords'       : 'HE chapter vocabulary terms',
    'has_headers'       : 'Markdown ### headers present',
    'has_bullets'       : 'Bullet / numbered list formatting',
    'is_english'        : 'Text appears to be in English',
    'questions_answered': 'Questions with substantive answers (>30 chars)',
    'trivial_answers'   : 'Answers that are just a number or "."',
    'copy_paste'        : 'Same text block detected in 2+ answers',
    'report_format'     : 'Consulting sub-sections (e.g. 2.1, 3.2)',
}

print("\n  12 signals computed for all students:")
for sig, desc in SIGNAL_DESCRIPTIONS.items():
    print(f"    {sig:<22}: {desc}")

print(f"\n  Signal matrix shape: {signals_df.shape}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 – SHOW SIGNAL DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 4 – Signal distributions across all students")
print("="*65)

print("\n  Raw signal values (all students):")
print(signals_df.describe().round(2).to_string())


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 – HARD RULE CLASSIFIERS (unambiguous signals)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 5 – Applying hard rule classifiers")
print("="*65)

print("""
  Hard rules (applied in order — first match wins if unambiguous):

  RULE A → Group 3 (Web Search)
    IF url_count >= 1
    RATIONALE: Only search users have URLs — textbook has none,
    NotebookLM never outputs URLs, LLMs don't paste URLs.

  RULE B → Group 2 (NotebookLM)
    IF cited_refs >= 3 OR fonti_refs >= 3
    RATIONALE: [1][2][3] citations and "le fonti" are NotebookLM's
    signature output style. No other tool produces these.

  RULE C → Group 1 (Textbook)
    IF trivial_answers >= 3
    RATIONALE: Answering 3+ questions with just a number or "."
    means the student couldn't find the answer — most consistent
    with textbook-only access.

  RULE D → Group 4 (LLM)
    IF is_english == 1 OR report_format == 1
    RATIONALE: Italian students writing in English, or producing
    numbered sub-section reports, are almost certainly using LLMs.
""")

def apply_hard_rules(row):
    """Returns (group, rule_applied) or (None, None) if no hard rule fires."""
    if row['url_count'] >= 1:
        return 3, 'RULE A: URL found in text'
    if row['cited_refs'] >= 3 or row['fonti_refs'] >= 3:
        return 2, 'RULE B: [n] citations or fonti_refs >= 3'
    if row['trivial_answers'] >= 3:
        return 1, 'RULE C: 3+ trivial answers (gave up)'
    if row['is_english'] == 1 or row['report_format'] == 1:
        return 4, 'RULE D: English text or consulting report format'
    return None, None

hard_results = {}
for case, row in signals_df.iterrows():
    group, rule = apply_hard_rules(row)
    hard_results[case] = {'hard_group': group, 'hard_rule': rule}

hard_df = pd.DataFrame(hard_results).T
classified_hard = hard_df[hard_df['hard_group'].notna()]
unclassified = hard_df[hard_df['hard_group'].isna()]

print(f"  Hard rules classified: {len(classified_hard)} students")
print(f"  Remaining for soft scoring: {len(unclassified)} students")

print("\n  Hard-rule classifications:")
for case, row in classified_hard.iterrows():
    print(f"    CASE {case:3.0f} → Group {int(row['hard_group'])}  [{row['hard_rule']}]")


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 – SOFT SCORING for ambiguous cases
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 6 – Soft scoring for ambiguous cases")
print("="*65)

print("""
  For students not classified by hard rules, we score each against
  group profiles using a weighted signal matrix.

  Scoring weights per group:

  Signal              G1(Textbook) G2(NotebookLM) G3(Search) G4(LLM)
  ─────────────────────────────────────────────────────────────────────
  word_count (high)      -1           +3             +1        +1
  he_keywords (>2)       +2           +3             -2        -1
  he_keywords (1-2)      +2           +1              0         0
  has_bullets             0           +2              0         0
  has_headers             0           +3              0        +2
  copy_paste             -1            0             +3        -1
  questions_answered(6)  -1           +2              0         0
  trivial_answers(1-2)   +2            0              0        -1
  report_format           0            0              0        +3
  is_english              0            0              0        +3

  Final group = argmax(score across G1, G2, G3, G4)
  Confidence = HIGH if winning score is 3+ points above runner-up
               MEDIUM otherwise
""")

def soft_score(row):
    """Score a student against each group profile. Returns dict of scores."""
    scores = {1: 0, 2: 0, 3: 0, 4: 0}

    wc = row['word_count']
    hk = row['he_keywords']
    qa = row['questions_answered']
    ta = row['trivial_answers']

    # word count signal
    if wc > 1500:
        scores[2] += 3
        scores[3] += 1
    elif wc > 600:
        scores[3] += 1
        scores[4] += 1
        scores[2] += 1
    else:
        scores[1] += 2

    # HE keywords
    if hk >= 5:
        scores[2] += 3
        scores[1] += 1
    elif hk >= 2:
        scores[1] += 2
        scores[2] += 1
    elif hk == 1:
        scores[1] += 1
    else:  # 0 HE keywords
        scores[3] += 1
        scores[4] += 1

    # bullets
    if row['has_bullets']:
        scores[2] += 2

    # headers
    if row['has_headers']:
        scores[2] += 2
        scores[4] += 1

    # copy paste
    if row['copy_paste']:
        scores[3] += 3
        scores[1] -= 1
        scores[4] -= 1

    # questions fully answered
    if qa == 6:
        scores[2] += 2
    elif qa <= 3:
        scores[1] += 2

    # trivial answers (1-2)
    if 1 <= ta <= 2:
        scores[1] += 2
        scores[4] -= 1

    # report format
    if row['report_format']:
        scores[4] += 3

    # english
    if row['is_english']:
        scores[4] += 3

    return scores

soft_results = {}
for case in unclassified.index:
    row = signals_df.loc[case]
    scores = soft_score(row)
    best_group = max(scores, key=scores.get)
    best_score = scores[best_group]
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1]
    confidence = 'HIGH' if margin >= 3 else 'MEDIUM'
    soft_results[case] = {
        'soft_group': best_group,
        'soft_scores': scores,
        'margin': margin,
        'confidence': confidence,
    }

print(f"  Soft scoring results ({len(soft_results)} students):")
for case, result in sorted(soft_results.items()):
    sc = result['soft_scores']
    print(f"    CASE {case:3.0f} → Group {result['soft_group']}  "
          f"[conf={result['confidence']}, margin={result['margin']:.0f}]  "
          f"scores: G1={sc[1]} G2={sc[2]} G3={sc[3]} G4={sc[4]}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 – MERGE HARD + SOFT → FINAL ASSIGNMENTS + MANUAL REVIEW NOTES
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 7 – Final assignments with manual review notes")
print("="*65)

# Manual review notes for each case (qualitative evidence read from text)
MANUAL_NOTES = {
    47: 'Uses "Sulla base delle fonti fornite"; systematic HE, no URLs',
    48: 'Inline [1][2][3] throughout; "co-evolvere [2,3]"; classic NotebookLM',
    49: 'Cites 2meet2biz.com, across.it, radar-academy.com with pasted excerpts',
    50: 'Invents creative falegnameria scenario; personal analogies; no source',
    51: 'General synthesis; no HE jargon; no URLs; fluent generative style',
    53: 'Uses Schumpeter + HE terms; Q3–Q6 answered as "3","4","5","6"',
    54: 'Centres on ceramic jewelry examples; personal tone; no source refs',
    55: 'Orecchiette-in-Puglia analogy; no HE/URLs; original synthesis',
    56: 'Written ENTIRELY in English; sophisticated original reasoning',
    57: 'Pastes Italian craft campaign article; Q5+Q6 have SAME paragraph (copy-paste)',
    58: 'Very short bullet answers; no HE jargon; no URLs; direct recall',
    59: 'Q2 answer is just a URL; references Google Ads/LinkedIn Ads',
    60: 'Numbered sub-sections 2.2, 3.1, 5.1; consulting report style; no HE/URLs',
    61: 'Uses "chi è / cosa e come fa" from Ch4; Q3–Q6 answered as "3","4","5","6"',
    62: 'Digital marketing focus; Q5: "Persone. Eliminare fame" (off-topic SDG copy)',
    63: '"secondo il modello descritto nei documenti"; systematic HE; no URLs',
    64: 'Full analysis of "Osacuca" (real Calabrian company); "Secondo me"',
    65: 'Well-structured professional Italian; no HE jargon; no URLs',
    66: '"Sebbene le fonti fornite non menzionino..." — transparent about source limits',
    68: 'Short personal examples (sarta; fattoria Pupo in Sila); no HE/URLs',
    69: 'Concise balanced; informally mentions "armonica" once; no URLs',
    70: 'Cites Schumpeter from chapter; Q6 answered as "."; numbered 1–6',
    71: 'Invents "Craft Passport" concept; "la via degli artigiani"; creative',
    72: 'Numbered [1][2][3] refs AND ### headers; "secondo le fonti"; full HE',
    74: 'Cites servizimultimediali.net, veniceoriginal.it, asvis.it with pasted text',
    75: '"homo sentimentalis" (from chapter); Calabria transport/spopolamento analysis',
    77: '"Imprenditore Armonico", "custode del creato"; Q5+Q6 answered as "."',
    78: 'Quotes "Venegoni" (external); succession planning (not in chapter)',
    79: 'Written in English; single-line answers; no HE/URLs',
    80: 'References "Feras" company; short personal opinions; no HE/URLs',
    82: 'Well-balanced professional Italian; no HE/URL references; general synthesis',
    84: 'All 6 answers are literally single numbers: "1","2","4","4","5","6"',
    85: 'All 6 answers are ONLY URLs — pure link list',
    87: '"double nexus","tre punte del diamante","le fonti"; systematic HE coverage',
}

# Merge results
final_assignments = []
for case in signals_df.index:
    case_int = int(case)
    sig_row = signals_df.loc[case]

    if case_int in [int(c) for c in classified_hard.index]:
        hrow = hard_df.loc[case]
        group = int(hrow['hard_group'])
        method = 'Hard rule'
        rule = hrow['hard_rule']
        confidence = 'HIGH'
        scores = {1: '-', 2: '-', 3: '-', 4: '-'}
    else:
        sresult = soft_results[case_int]
        group = sresult['soft_group']
        method = 'Soft score'
        rule = f"Margin={sresult['margin']:.0f} over runner-up"
        confidence = sresult['confidence']
        scores = sresult['soft_scores']

    final_assignments.append({
        'CASE'             : case_int,
        'Group'            : group,
        'Method'           : method,
        'Rule_Applied'     : rule,
        'Confidence'       : confidence,
        'Score_G1'         : scores[1],
        'Score_G2'         : scores[2],
        'Score_G3'         : scores[3],
        'Score_G4'         : scores[4],
        'word_count'       : int(sig_row['word_count']),
        'url_count'        : int(sig_row['url_count']),
        'cited_refs'       : int(sig_row['cited_refs']),
        'fonti_refs'       : int(sig_row['fonti_refs']),
        'he_keywords'      : int(sig_row['he_keywords']),
        'trivial_answers'  : int(sig_row['trivial_answers']),
        'is_english'       : int(sig_row['is_english']),
        'report_format'    : int(sig_row['report_format']),
        'copy_paste'       : int(sig_row['copy_paste']),
        'Manual_Notes'     : MANUAL_NOTES.get(case_int, ''),
    })

final_df = pd.DataFrame(final_assignments).sort_values('CASE').reset_index(drop=True)

GROUP_LABELS = {
    1: 'Knowledge Learning (Textbook)',
    2: 'Information Extraction (NotebookLM)',
    3: 'Knowledge Discovery (Web Search)',
    4: 'Interaction (LLM)',
}
final_df['Group_Label'] = final_df['Group'].map(GROUP_LABELS)

final_df.to_csv(os.path.join(OUTPUT_DIR, 'step7_final_classifications.csv'), index=False)
print("  Saved: step7_final_classifications.csv\n")

# Print the full decision table
print("  FULL CLASSIFICATION DECISION TABLE:")
print(f"  {'CASE':>4}  {'Group':>5}  {'Conf':>6}  {'Method':>10}  {'Rule / Evidence'}")
print("  " + "-"*80)
for _, row in final_df.iterrows():
    rule_short = str(row['Rule_Applied'])[:50]
    print(f"  {int(row['CASE']):4d}  {int(row['Group']):5d}  {row['Confidence']:>6}  "
          f"{row['Method']:>10}  {rule_short}")

# Summary
print("\n  FINAL GROUP DISTRIBUTION:")
gc = final_df['Group'].value_counts().sort_index()
for g, n in gc.items():
    nh = (final_df[final_df['Group']==g]['Confidence']=='HIGH').sum()
    nm = (final_df[final_df['Group']==g]['Confidence']=='MEDIUM').sum()
    print(f"    Group {g} ({GROUP_LABELS[g]}): n={n}  (HIGH={nh}, MEDIUM={nm})")


# ════════════════════════════════════════════════════════════════════════════
# STEP 8 – PER-STUDENT EVIDENCE REPORT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 8 – Per-student evidence report")
print("="*65)

GROUP_COLORS = {1: '#534AB7', 2: '#0F6E56', 3: '#993C1D', 4: '#185FA5'}

for g in [1, 2, 3, 4]:
    group_cases = final_df[final_df['Group'] == g]
    print(f"\n  ── Group {g}: {GROUP_LABELS[g]} (n={len(group_cases)}) ──")
    for _, row in group_cases.iterrows():
        print(f"\n    CASE {int(row['CASE'])}  [{row['Confidence']}]  "
              f"via {row['Method']}")
        print(f"      Signals: words={row['word_count']}, URLs={row['url_count']}, "
              f"[n]refs={row['cited_refs']}, fonti={row['fonti_refs']}, "
              f"HE_terms={row['he_keywords']}, trivial={row['trivial_answers']}, "
              f"english={row['is_english']}, report={row['report_format']}")
        print(f"      Evidence: {row['Manual_Notes']}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 9 – VISUALISE THE CLASSIFICATION PROCESS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 9 – Visualising classification signals and results")
print("="*65)

# V1: Signal matrix heatmap — each row = student, each col = signal
# ─────────────────────────────────────────────────────────────────
sig_display_cols = [
    'url_count', 'cited_refs', 'fonti_refs', 'he_keywords',
    'has_headers', 'has_bullets', 'trivial_answers',
    'is_english', 'copy_paste', 'report_format',
]
sig_labels = [
    'URLs', '[n] Citations', 'Fonti refs', 'HE keywords',
    'MD Headers', 'Bullets', 'Trivial ans.', 'English',
    'Copy-paste', 'Report fmt',
]

hm_data = signals_df[sig_display_cols].copy()
# Clip for display
hm_data['url_count']    = hm_data['url_count'].clip(0, 5)
hm_data['cited_refs']   = hm_data['cited_refs'].clip(0, 15)
hm_data['fonti_refs']   = hm_data['fonti_refs'].clip(0, 12)
hm_data['he_keywords']  = hm_data['he_keywords'].clip(0, 15)

hm_data.columns = sig_labels

# Sort by assigned group then case
case_to_group = final_df.set_index('CASE')['Group'].to_dict()
hm_data['_group'] = hm_data.index.map(lambda c: case_to_group.get(int(c), 0))
hm_data = hm_data.sort_values('_group').drop(columns='_group')

# Build row color bar
row_colors = [GROUP_COLORS[case_to_group.get(int(c), 1)] for c in hm_data.index]

fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(
    hm_data.T,
    cmap='YlOrRd', annot=True, fmt='.0f',
    linewidths=0.3, linecolor='white',
    ax=ax,
    cbar_kws={'label': 'Signal value', 'shrink': 0.6}
)

# Colour the case labels by group
for i, (tick, case) in enumerate(zip(ax.get_xticklabels(), hm_data.index)):
    g = case_to_group.get(int(case), 1)
    tick.set_color(GROUP_COLORS[g])
    tick.set_fontweight('bold')
    tick.set_fontsize(8)

ax.set_xticklabels([f'C{int(c)}' for c in hm_data.index], rotation=45, ha='right')
ax.set_yticklabels(sig_labels, rotation=0, fontsize=10)
ax.set_title('Classification Signal Matrix\n(students sorted by assigned group)',
             fontsize=13, pad=12)

legend_patches = [
    mpatches.Patch(color=GROUP_COLORS[g],
                   label=f'G{g}: {GROUP_LABELS[g].split("(")[0].strip()}')
    for g in [1, 2, 3, 4]
]
ax.legend(handles=legend_patches, loc='upper right',
          bbox_to_anchor=(1.18, 1.05), fontsize=9, title='Group')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step9a_signal_matrix.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: step9a_signal_matrix.png")


# V2: Scatter — URL count vs HE keywords (separates G2 and G3 perfectly)
# ────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for case_val, row in signals_df.iterrows():
    g = case_to_group.get(int(case_val), 1)
    jitter_x = np.random.uniform(-0.1, 0.1)
    jitter_y = np.random.uniform(-0.1, 0.1)
    axes[0].scatter(
        row['url_count'] + jitter_x,
        row['he_keywords'] + jitter_y,
        color=GROUP_COLORS[g], s=70, alpha=0.8, edgecolors='white', linewidth=0.5
    )
    axes[0].annotate(f'C{int(case_val)}', (row['url_count'] + jitter_x,
                                            row['he_keywords'] + jitter_y),
                     fontsize=6, color=GROUP_COLORS[g], alpha=0.8)

axes[0].set_xlabel('URL count in answers', fontsize=11)
axes[0].set_ylabel('HE chapter keywords count', fontsize=11)
axes[0].set_title('URLs vs HE Keywords\n(G3 top-left, G2 top-right area)', fontsize=11)
axes[0].axvline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
axes[0].axhline(2, color='gray', linestyle='--', alpha=0.4, linewidth=1)
for g in [1, 2, 3, 4]:
    axes[0].scatter([], [], color=GROUP_COLORS[g], label=f'G{g}', s=50)
axes[0].legend(fontsize=9)


# V3: [n] citations vs fonti_refs (isolates G2 perfectly)
for case_val, row in signals_df.iterrows():
    g = case_to_group.get(int(case_val), 1)
    jitter_x = np.random.uniform(-0.15, 0.15)
    jitter_y = np.random.uniform(-0.15, 0.15)
    axes[1].scatter(
        row['cited_refs'] + jitter_x,
        row['fonti_refs'] + jitter_y,
        color=GROUP_COLORS[g], s=70, alpha=0.8, edgecolors='white', linewidth=0.5
    )
    axes[1].annotate(f'C{int(case_val)}',
                     (row['cited_refs'] + jitter_x, row['fonti_refs'] + jitter_y),
                     fontsize=6, color=GROUP_COLORS[g], alpha=0.8)

axes[1].set_xlabel('[n] inline citations count', fontsize=11)
axes[1].set_ylabel('"Le fonti / i documenti" phrase count', fontsize=11)
axes[1].set_title('[n] Citations vs Fonti Refs\n(G2 = NotebookLM occupies top-right)', fontsize=11)
axes[1].axvline(2.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
axes[1].axhline(2.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
for g in [1, 2, 3, 4]:
    axes[1].scatter([], [], color=GROUP_COLORS[g], label=f'G{g}', s=50)
axes[1].legend(fontsize=9)

plt.suptitle('Signal Scatter Plots for Group Separation', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step9b_signal_scatterplots.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: step9b_signal_scatterplots.png")


# V4: Group means of all signals (bar chart — shows how groups differ)
# ──────────────────────────────────────────────────────────────────────
signals_df_copy = signals_df.copy()
signals_df_copy['Group'] = signals_df_copy.index.map(
    lambda c: case_to_group.get(int(c), None)
)
signals_df_copy = signals_df_copy.dropna(subset=['Group'])
signals_df_copy['Group'] = signals_df_copy['Group'].astype(int)

group_signal_means = signals_df_copy.groupby('Group')[sig_display_cols].mean()

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
axes = axes.flatten()

for i, (col, label) in enumerate(zip(sig_display_cols, sig_labels)):
    ax = axes[i]
    vals = [group_signal_means.loc[g, col] if g in group_signal_means.index else 0
            for g in [1, 2, 3, 4]]
    bars = ax.bar([f'G{g}' for g in [1, 2, 3, 4]], vals,
                  color=[GROUP_COLORS[g] for g in [1, 2, 3, 4]], alpha=0.82)
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean value', fontsize=8)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.03,
                    f'{val:.1f}', ha='center', fontsize=8)

legend_patches = [
    mpatches.Patch(color=GROUP_COLORS[g],
                   label=f'G{g}: {GROUP_LABELS[g].split("(")[0].strip()}')
    for g in [1, 2, 3, 4]
]
fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, 0.0))
fig.suptitle('Mean Signal Values by Group — Shows How Groups Are Separated',
             fontsize=13)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(os.path.join(OUTPUT_DIR, 'step9c_group_signal_means.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: step9c_group_signal_means.png")


# V5: Classification flow summary (counts per step)
# ──────────────────────────────────────────────────
hard_counts = final_df[final_df['Method']=='Hard rule']['Group'].value_counts().sort_index()
soft_counts = final_df[final_df['Method']=='Soft score']['Group'].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

def make_bar(ax, counts, title):
    bars = ax.bar([f'G{g}' for g in counts.index], counts.values,
                  color=[GROUP_COLORS[g] for g in counts.index], alpha=0.82)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of students', fontsize=10)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha='center', fontsize=11, fontweight='bold')

make_bar(axes[0], hard_counts,
         f'Hard Rule Classifications\n({len(classified_hard)} students — unambiguous)')
make_bar(axes[1], soft_counts,
         f'Soft Score Classifications\n({len(soft_results)} students — soft evidence)')

for ax in axes:
    ax.set_xticklabels([f'G{g}\n{GROUP_LABELS[g].split("(")[0]}' for g in [1,2,3,4]],
                       fontsize=8)

plt.suptitle('Students Classified by Method', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step9d_classification_flow.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: step9d_classification_flow.png")


# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("CLASSIFICATION COMPLETE – Output files:")
print("="*65)
files = [
    ('step7_final_classifications.csv',
     'Full decision table: group, method, rule, signals, notes'),
    ('step9a_signal_matrix.png',
     'Heatmap of all 10 signals for all 34 students'),
    ('step9b_signal_scatterplots.png',
     'URL vs HE keywords & [n]refs vs Fonti — group separation plots'),
    ('step9c_group_signal_means.png',
     'Bar charts of mean signal per group for each of 10 signals'),
    ('step9d_classification_flow.png',
     'How many students classified by hard rules vs soft scoring'),
]
for fname, desc in files:
    print(f"  {fname:<42} ← {desc}")

print("""
HOW TO READ THE METHODOLOGY:
─────────────────────────────────────────────────────────
STEP 1: We extracted all 6 open-text answers per student and
        concatenated them into one text blob for analysis.

STEP 2: We defined 4 "fingerprints" — the linguistic patterns
        each tool leaves in student writing.

STEP 3: We computed 12 measurable signals from each student's text
        (URLs, [n] citations, "fonti" phrases, HE keywords, English
        detection, trivial answers, copy-paste detection, etc.)

STEP 4: We inspected the distribution of signals across students.

STEP 5: We applied 4 hard rules that fire on unambiguous signals:
        — Any URL → Group 3 (web search)
        — [n] citations or fonti refs ≥ 3 → Group 2 (NotebookLM)
        — 3+ trivial answers → Group 1 (textbook)
        — English or report format → Group 4 (LLM)

STEP 6: For remaining ambiguous students, we scored each against all
        4 group profiles using weighted signal matching.

STEP 7: We merged decisions, flagged confidence levels, and added
        qualitative reading notes for each student.

STEP 8: We printed a full per-student evidence report.

STEP 9: We visualised the process with 4 charts.
""")