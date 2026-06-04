

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from collections import Counter
import os

# ── colour palette consistent across all plots ─────────────────────────────
COLORS = {
    'A0': '#534AB7',  # purple  – Motivation
    'A1': '#0F6E56',  # teal    – Well-being
    'A2': '#993C1D',  # coral   – Ethics
    'A3': '#185FA5',  # blue    – Relationships
    'A4': '#3B6D11',  # green   – Innovation
    'A5': '#BA7517',  # amber   – Social function
    'A6': '#993556',  # pink    – Self-awareness
}
SECTION_LABELS = {
    'A0': 'Entrepreneurial\nMotivation',
    'A1': 'Personal\nWell-being',
    'A2': 'Ethics &\nResponsibility',
    'A3': 'Relationships &\nStakeholders',
    'A4': 'Innovation &\nFuture',
    'A5': 'Social Function\nof Business',
    'A6': 'Entrepreneurial\nSelf-awareness',
}
HE_PILLARS = {
    'A0': 'Entrepreneurial orientation',
    'A1': 'Salus salutis',
    'A2': 'Homo humanitas',
    'A3': 'Homo humanitas',
    'A4': 'Homo humus',
    'A5': 'Homo humanitas',
    'A6': 'Entrepreneurial orientation',
}

OUTPUT_DIR = '.'   # change this if you want outputs in a subfolder


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 – LOAD & CLEAN DATA
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1 – Loading and cleaning data")
print("="*60)

df_raw = pd.read_excel('data_unical_2026-05-12_15-12.xlsx', header=None)

# Row 0 = column codes, Row 1 = labels, Rows 2+ = data
df = df_raw.iloc[2:].copy()
df.columns = df_raw.iloc[0].tolist()
df = df.reset_index(drop=True)

# Convert Likert columns to numeric
LIKERT_COLS = [
    'A001','A002','A003',
    'A101','A102','A103',
    'A201','A202','A203',
    'A301','A302','A303',
    'A401','A402','A403',
    'A501','A502','A503',
    'A601','A602',
]
for col in LIKERT_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['TIME_SUM'] = pd.to_numeric(df['TIME_SUM'], errors='coerce')
df['FINISHED']  = pd.to_numeric(df['FINISHED'],  errors='coerce')
df['MISSING']   = pd.to_numeric(df['MISSING'],   errors='coerce')

# Filter: completed responses only
df_all   = df.copy()
df       = df[df['FINISHED'] == 1].copy().reset_index(drop=True)

# Drop rows missing any Likert item
df_clean = df.dropna(subset=LIKERT_COLS).copy().reset_index(drop=True)

print(f"  Total records in file  : {len(df_all)}")
print(f"  Completed (FINISHED=1) : {len(df)}")
print(f"  Full Likert data       : {len(df_clean)}")

# ── Decode demographics ──────────────────────────────────────────────────
# PI01 Age group  : 1 = Under 30,  2 = 30 or older
# PI02 Gender     : 1 = Male,      2 = Female
# PI04 Education  : 1 = Primary, 2 = Secondary/High school, 3 = Bachelor,
#                   4 = Master, 5 = PhD
df_clean['Age_label'] = df_clean['PI01'].map({1: 'Under 30', 2: '30+'})
df_clean['Gender_label'] = df_clean['PI02'].map({1: 'Male', 2: 'Female'})
df_clean['Education_label'] = df_clean['PI04'].map({
    1: 'Primary', 2: 'High school', 3: 'Bachelor', 4: 'Master', 5: 'PhD'
})

print("\n  Demographics (completed responses):")
print(f"    Age under 30: {(df_clean['PI01']==1).sum()}  |  30+: {(df_clean['PI01']==2).sum()}")
print(f"    Male: {(df_clean['PI02']==1).sum()}  |  Female: {(df_clean['PI02']==2).sum()}")
print(f"    Education: {df_clean['Education_label'].value_counts().to_dict()}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 – COMPUTE SECTION SCORES AND HE INDEX
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2 – Computing section scores and HE composite index")
print("="*60)

SECTION_ITEMS = {
    'A0': ['A001','A002','A003'],
    'A1': ['A101','A102','A103'],
    'A2': ['A201','A202','A203'],
    'A3': ['A301','A302','A303'],
    'A4': ['A401','A402','A403'],
    'A5': ['A501','A502','A503'],
    'A6': ['A601','A602'],
}
SCORE_COLS = [f'{s}_score' for s in SECTION_ITEMS]

for sec, items in SECTION_ITEMS.items():
    df_clean[f'{sec}_score'] = df_clean[items].mean(axis=1)

# Overall HE composite (mean of all 20 items, scale 1–4)
df_clean['HE_index'] = df_clean[LIKERT_COLS].mean(axis=1)

# Pillar scores (grouped by HE theory)
df_clean['Salus_score']    = df_clean['A1_score']
df_clean['Humanitas_score']= df_clean[['A2_score','A3_score','A5_score']].mean(axis=1)
df_clean['Humus_score']    = df_clean['A4_score']

print(f"  HE index range: {df_clean['HE_index'].min():.2f} – {df_clean['HE_index'].max():.2f}")
print(f"  HE index mean : {df_clean['HE_index'].mean():.2f}  SD={df_clean['HE_index'].std():.2f}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 – DESCRIPTIVE STATISTICS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3 – Descriptive statistics")
print("="*60)

desc_items = df_clean[LIKERT_COLS].describe().T
desc_items['skewness'] = df_clean[LIKERT_COLS].skew()
desc_items['kurtosis'] = df_clean[LIKERT_COLS].kurtosis()
print("\nItem-level descriptives:")
print(desc_items[['count','mean','std','min','25%','50%','75%','max','skewness']].round(2).to_string())

desc_sections = df_clean[SCORE_COLS + ['HE_index']].describe().T
print("\nSection-level descriptives:")
print(desc_sections[['count','mean','std','min','50%','max']].round(2).to_string())

desc_items.to_csv(os.path.join(OUTPUT_DIR, 'step3_item_descriptives.csv'))
desc_sections.to_csv(os.path.join(OUTPUT_DIR, 'step3_section_descriptives.csv'))
print("\n  Saved: step3_item_descriptives.csv, step3_section_descriptives.csv")


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 – RELIABILITY (CRONBACH'S ALPHA) PER SECTION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 4 – Reliability (Cronbach's alpha)")
print("="*60)

def cronbach_alpha(df_items):
    """Compute Cronbach's alpha from a DataFrame of items."""
    df_items = df_items.dropna()
    n = df_items.shape[1]
    if n < 2:
        return np.nan
    item_vars = df_items.var(axis=0, ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    return (n / (n - 1)) * (1 - item_vars.sum() / total_var)

alpha_results = []
for sec, items in SECTION_ITEMS.items():
    if len(items) < 2:
        alpha = np.nan
        interpretation = 'N/A (single item)'
    else:
        alpha = cronbach_alpha(df_clean[items])
        if alpha >= 0.9:   interpretation = 'Excellent'
        elif alpha >= 0.8: interpretation = 'Good'
        elif alpha >= 0.7: interpretation = 'Acceptable'
        elif alpha >= 0.6: interpretation = 'Questionable'
        elif alpha >= 0.5: interpretation = 'Poor'
        else:              interpretation = 'Unacceptable'
    alpha_results.append({
        'Section': sec,
        'Label': SECTION_LABELS[sec].replace('\n',' '),
        'Items': len(items),
        'Alpha': round(alpha, 3),
        'Interpretation': interpretation
    })
    print(f"  {sec} ({SECTION_LABELS[sec].replace(chr(10),' ')}): α={alpha:.3f}  →  {interpretation}")

alpha_df = pd.DataFrame(alpha_results)
alpha_df.to_csv(os.path.join(OUTPUT_DIR, 'step4_cronbach_alpha.csv'), index=False)
print("\n  Saved: step4_cronbach_alpha.csv")
print("\n  NOTE: α < 0.70 means items in that section don't measure the same")
print("  construct reliably. Interpret those section scores with caution.")


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 – DEMOGRAPHIC ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 5 – Demographic analysis")
print("="*60)

demo_results = []

for demo_col, demo_label in [('Gender_label', 'Gender'), ('Age_label', 'Age group')]:
    groups = df_clean[demo_col].dropna().unique()
    print(f"\n  --- {demo_label} ---")
    for score_col in SCORE_COLS + ['HE_index']:
        group_data = [df_clean[df_clean[demo_col]==g][score_col].dropna() for g in groups]
        if len(group_data) == 2 and all(len(g) >= 3 for g in group_data):
            stat, p = stats.mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')
            means = {g: group_data[i].mean() for i, g in enumerate(groups)}
            sig = '**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.')
            print(f"    {score_col}: {means}  p={p:.3f} {sig}")
            demo_results.append({
                'Variable': demo_label, 'Score': score_col,
                **{f'Mean_{g}': round(v, 2) for g, v in means.items()},
                'Mann-Whitney U': round(stat, 2), 'p-value': round(p, 3), 'Sig': sig
            })

demo_df = pd.DataFrame(demo_results)
demo_df.to_csv(os.path.join(OUTPUT_DIR, 'step5_demographic_analysis.csv'), index=False)
print("\n  Saved: step5_demographic_analysis.csv")
print("  (* p<0.05  ** p<0.01  n.s. = not significant)")


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 – CORRELATION MATRIX BETWEEN SECTIONS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 6 – Correlation matrix (Spearman, ordinal data)")
print("="*60)

corr_data = df_clean[SCORE_COLS].copy()
corr_data.columns = [SECTION_LABELS[s.replace('_score','')] for s in SCORE_COLS]
corr_matrix = corr_data.corr(method='spearman')

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # show lower triangle
sns.heatmap(
    corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
    vmin=-1, vmax=1, center=0, linewidths=0.5,
    ax=ax, cbar_kws={'label': 'Spearman ρ'}
)
ax.set_title('Spearman Correlation Between Section Scores', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step6_correlation_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()

corr_matrix.to_csv(os.path.join(OUTPUT_DIR, 'step6_correlation_matrix.csv'))
print("  Saved: step6_correlation_matrix.png, step6_correlation_matrix.csv")

# Print strong correlations
print("\n  Notable correlations (|ρ| > 0.50):")
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i,j]
        if abs(r) > 0.50:
            print(f"    {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: ρ={r:.2f}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 – RADAR CHART (overall + by gender + by age)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 7 – Radar/spider charts")
print("="*60)

def radar_chart(data_dict, title, filename, scale_max=4):
    """
    data_dict: {'Group label': [7 section score values], ...}
    Values must correspond to SECTION_ITEMS order: A0–A6
    """
    labels = [SECTION_LABELS[s] for s in SECTION_ITEMS.keys()]
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the loop

    palette = ['#534AB7','#0F6E56','#993C1D','#185FA5']
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for idx, (group_name, values) in enumerate(data_dict.items()):
        values_plot = list(values) + [values[0]]
        color = palette[idx % len(palette)]
        ax.plot(angles, values_plot, 'o-', linewidth=2, label=group_name, color=color)
        ax.fill(angles, values_plot, alpha=0.10, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(1, scale_max)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['1\n(least\nharmonic)', '2', '3', '4\n(fully\nharmonic)'], size=7)
    ax.set_title(title, size=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

# Overall profile
overall_scores = [df_clean[f'{s}_score'].mean() for s in SECTION_ITEMS.keys()]
radar_chart({'All students': overall_scores}, 'HE Profile – Overall Sample', 'step7a_radar_overall.png')

# By gender
gender_data = {}
for g in ['Male','Female']:
    sub = df_clean[df_clean['Gender_label']==g]
    if len(sub) >= 3:
        gender_data[g] = [sub[f'{s}_score'].mean() for s in SECTION_ITEMS.keys()]
if len(gender_data) == 2:
    radar_chart(gender_data, 'HE Profile by Gender', 'step7b_radar_gender.png')

# By age
age_data = {}
for a in ['Under 30','30+']:
    sub = df_clean[df_clean['Age_label']==a]
    if len(sub) >= 3:
        age_data[a] = [sub[f'{s}_score'].mean() for s in SECTION_ITEMS.keys()]
if len(age_data) >= 2:
    radar_chart(age_data, 'HE Profile by Age Group', 'step7c_radar_age.png')


# ════════════════════════════════════════════════════════════════════════════
# STEP 8 – ITEM-LEVEL RESPONSE DISTRIBUTIONS (stacked bar)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 8 – Item response distribution charts")
print("="*60)

RESPONSE_LABELS = {
    1: '1 – Least harmonic',
    2: '2 – Transitional',
    3: '3 – Socially responsible',
    4: '4 – Fully harmonic',
}
RESPONSE_COLORS = ['#F09595','#FAC775','#9FE1CB','#1D9E75']

fig, axes = plt.subplots(4, 5, figsize=(18, 13))
axes = axes.flatten()

for idx, col in enumerate(LIKERT_COLS):
    ax = axes[idx]
    counts = df_clean[col].value_counts().sort_index()
    pcts   = (counts / len(df_clean) * 100).reindex([1,2,3,4], fill_value=0)
    bars = ax.bar([1,2,3,4], pcts.values, color=RESPONSE_COLORS, edgecolor='white', linewidth=0.5)
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.set_xlabel('Response (1–4)', fontsize=8)
    ax.set_ylabel('% respondents', fontsize=8)
    ax.set_xticks([1,2,3,4])
    ax.set_ylim(0, 70)
    for bar, pct in zip(bars, pcts.values):
        if pct > 3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{pct:.0f}%', ha='center', va='bottom', fontsize=7)

# Hide unused subplots
for idx in range(len(LIKERT_COLS), len(axes)):
    axes[idx].set_visible(False)

legend_patches = [mpatches.Patch(color=RESPONSE_COLORS[i], label=RESPONSE_LABELS[i+1]) for i in range(4)]
fig.legend(handles=legend_patches, loc='lower right', fontsize=9, ncol=2)
fig.suptitle('Response Distribution per Likert Item (% of respondents)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step8_item_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: step8_item_distributions.png")

# Section-level mean bar chart
fig, ax = plt.subplots(figsize=(10, 5))
sections = list(SECTION_ITEMS.keys())
means    = [df_clean[f'{s}_score'].mean() for s in sections]
sds      = [df_clean[f'{s}_score'].std()  for s in sections]
colors   = [COLORS[s] for s in sections]
labels   = [SECTION_LABELS[s].replace('\n',' ') for s in sections]

bars = ax.bar(labels, means, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
ax.errorbar(labels, means, yerr=sds, fmt='none', color='#444', capsize=4, linewidth=1.2)
ax.axhline(y=2.5, color='gray', linestyle='--', linewidth=1, label='Midpoint (2.5)')
ax.set_ylim(1, 4)
ax.set_ylabel('Mean score (1–4)', fontsize=11)
ax.set_title('Section Score Means with Standard Deviation', fontsize=13)
ax.tick_params(axis='x', labelsize=9)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{mean:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step8_section_means.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: step8_section_means.png")


# ════════════════════════════════════════════════════════════════════════════
# STEP 9 – RESPONSE TIME ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 9 – Response time analysis")
print("="*60)

df_time = df_clean.dropna(subset=['TIME_SUM']).copy()
df_time = df_time[df_time['TIME_SUM'] > 60]  # remove outliers under 1 min

print(f"  Valid TIME_SUM records: {len(df_time)}")
print(f"  Mean completion time: {df_time['TIME_SUM'].mean():.0f}s  "
      f"({df_time['TIME_SUM'].mean()/60:.1f} min)")
print(f"  Median: {df_time['TIME_SUM'].median():.0f}s")
print(f"  SD: {df_time['TIME_SUM'].std():.0f}s")

# Correlation: time spent vs HE_index
r, p = stats.spearmanr(df_time['TIME_SUM'], df_time['HE_index'])
print(f"\n  Spearman ρ (TIME_SUM vs HE_index): {r:.3f}  p={p:.3f}")
sig = '(significant)' if p < 0.05 else '(not significant)'
print(f"  {sig}")
print("  → Longer completion time may indicate more careful, reflective answering")

# Time distribution chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df_time['TIME_SUM'] / 60, bins=10, color='#534AB7', alpha=0.75, edgecolor='white')
axes[0].axvline(df_time['TIME_SUM'].mean()/60, color='#993C1D', linewidth=2, linestyle='--',
                label=f"Mean: {df_time['TIME_SUM'].mean()/60:.1f} min")
axes[0].set_xlabel('Completion time (minutes)', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_title('Distribution of Completion Times', fontsize=12)
axes[0].legend()

axes[1].scatter(df_time['TIME_SUM']/60, df_time['HE_index'],
                color='#0F6E56', alpha=0.65, s=60, edgecolors='white')
m, b = np.polyfit(df_time['TIME_SUM']/60, df_time['HE_index'], 1)
x_line = np.linspace(df_time['TIME_SUM'].min()/60, df_time['TIME_SUM'].max()/60, 100)
axes[1].plot(x_line, m*x_line + b, color='#993C1D', linewidth=1.5, linestyle='--',
             label=f'Trend  ρ={r:.2f}')
axes[1].set_xlabel('Completion time (minutes)', fontsize=11)
axes[1].set_ylabel('HE composite index', fontsize=11)
axes[1].set_title('Completion Time vs HE Index', fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step9_response_time.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: step9_response_time.png")


# ════════════════════════════════════════════════════════════════════════════
# STEP 10 – FIVE-WORD FREQUENCY ANALYSIS (A710)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 10 – Five-words frequency analysis (A710)")
print("="*60)

word_cols = ['A710x01','A710x02','A710x03','A710x04','A710x05']
all_words = []
for col in word_cols:
    words = df_clean[col].fillna('').astype(str)
    words = words[words.str.strip().str.len() > 0]
    all_words.extend([w.strip().lower().capitalize() for w in words if w.strip() not in ['','nan']])

word_counts = Counter(all_words)
top_words = word_counts.most_common(20)
print("\n  Top 20 words students associate with the harmonic entrepreneur:")
for word, count in top_words:
    print(f"    {word:<30} {count}")

word_df = pd.DataFrame(top_words, columns=['Word','Count'])
word_df.to_csv(os.path.join(OUTPUT_DIR, 'step10_word_frequency.csv'), index=False)

# Bar chart
fig, ax = plt.subplots(figsize=(10, 6))
words_plot = [w for w, _ in top_words[:15]]
counts_plot = [c for _, c in top_words[:15]]
bars = ax.barh(words_plot[::-1], counts_plot[::-1], color='#534AB7', alpha=0.8)
ax.set_xlabel('Frequency', fontsize=11)
ax.set_title('Top 15 Words Students Associate with the Harmonic Entrepreneur', fontsize=12)
for bar, count in zip(bars, counts_plot[::-1]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            str(count), va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step10_word_frequency.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: step10_word_frequency.png, step10_word_frequency.csv")


# ════════════════════════════════════════════════════════════════════════════
# STEP 11 – OPEN-TEXT QUALITY METRICS (A7 section)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 11 – Open-text quality metrics (A7 craftsmanship responses)")
print("="*60)

TEXT_COLS = {
    'A701_01': 'Innovate a craft',
    'A702_01': 'Elements of territory',
    'A703_01': 'Customer value',
    'A704_01': 'Sustainability',
    'A705_01': '2030 Agenda',
    'A706_01': 'Local economy impact',
}

print("""
  These metrics are LANGUAGE-AGNOSTIC proxies for response quality.
  They do NOT require understanding Italian. They measure:
    - Length           → effort/elaboration
    - Word count       → elaboration
    - Unique words     → vocabulary richness
    - Type-token ratio → lexical diversity (unique/total words)
    - Bullet markers   → structured formatting (*, -, •, numbered)
    - Citation markers → reference to sources ([1], [2], etc.)
    - Avg word length  → proxy for domain vocabulary sophistication
""")

text_metrics = []
for col, label in TEXT_COLS.items():
    series = df_clean[col].fillna('').astype(str)
    valid  = series[series.str.strip().str.len() > 20]
    
    for idx, text in valid.items():
        words = text.split()
        unique_words = set(w.lower() for w in words)
        has_bullets  = int(any(m in text for m in ['* ', '- ', '• ', '1.', '2.', '3.']))
        has_citations= int('[' in text and ']' in text)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        ttr = len(unique_words) / len(words) if words else 0
        
        text_metrics.append({
            'CASE'        : df_clean.loc[idx, 'CASE'],
            'Question'    : label,
            'Char_length' : len(text),
            'Word_count'  : len(words),
            'Unique_words': len(unique_words),
            'TTR'         : round(ttr, 3),
            'Has_bullets' : has_bullets,
            'Has_citations': has_citations,
            'Avg_word_len': round(avg_word_len, 2),
            'HE_index'    : df_clean.loc[idx, 'HE_index'],
            'Gender'      : df_clean.loc[idx, 'Gender_label'],
            'Age'         : df_clean.loc[idx, 'Age_label'],
        })

text_df = pd.DataFrame(text_metrics)
print("  Summary of text quality metrics across all A7 questions:")
print(text_df[['Char_length','Word_count','Unique_words','TTR',
               'Has_bullets','Has_citations','Avg_word_len']].describe().round(2).to_string())

# Correlate text quality with HE index
print("\n  Spearman correlations: text quality vs HE index (per response):")
for metric in ['Char_length','Word_count','TTR','Has_bullets','Has_citations','Avg_word_len']:
    sub = text_df[['HE_index',metric]].dropna()
    if len(sub) > 5:
        r, p = stats.spearmanr(sub['HE_index'], sub[metric])
        sig = '*' if p < 0.05 else ''
        print(f"    {metric:<20}: ρ={r:+.3f}  p={p:.3f} {sig}")

# Per-question summary
print("\n  Per-question mean word count:")
for q in TEXT_COLS.values():
    sub = text_df[text_df['Question']==q]
    print(f"    {q:<25}: {sub['Word_count'].mean():.0f} words  "
          f"TTR={sub['TTR'].mean():.3f}  "
          f"Bullets={sub['Has_bullets'].mean()*100:.0f}%  "
          f"Citations={sub['Has_citations'].mean()*100:.0f}%")

text_df.to_csv(os.path.join(OUTPUT_DIR, 'step11_text_quality_metrics.csv'), index=False)

# Visual: word count by question
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

q_means = text_df.groupby('Question')['Word_count'].mean().sort_values()
axes[0].barh(q_means.index, q_means.values, color='#185FA5', alpha=0.8)
axes[0].set_xlabel('Mean word count', fontsize=11)
axes[0].set_title('A7: Mean response length by question', fontsize=11)

text_df.boxplot(column='Word_count', by='Question', ax=axes[1], rot=30)
axes[1].set_title('Word count distribution per question', fontsize=11)
axes[1].set_xlabel('')
plt.suptitle('')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step11_text_quality.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: step11_text_quality.png, step11_text_quality_metrics.csv")


# ════════════════════════════════════════════════════════════════════════════
# STEP 12 – HE LEVEL SEGMENTATION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 12 – HE Level Segmentation")
print("="*60)

print("""
  Since groups are unknown, we segment students by their HE composite score
  into 3 levels aligned with the theoretical framework:
    Level 1 (Nascent HE) : HE_index < 2.5   → profit-first orientation
    Level 2 (Developing) : 2.5 ≤ HE_index < 3.25  → transitional
    Level 3 (Advanced HE): HE_index ≥ 3.25  → holistic harmonic thinking
""")

def he_level(score):
    if score < 2.5:    return 'Level 1 – Nascent HE'
    elif score < 3.25: return 'Level 2 – Developing HE'
    else:              return 'Level 3 – Advanced HE'

df_clean['HE_level'] = df_clean['HE_index'].apply(he_level)
print("  Distribution:")
print(df_clean['HE_level'].value_counts().to_string())

level_profile = df_clean.groupby('HE_level')[SCORE_COLS + ['HE_index']].mean().round(2)
print("\n  Section means per HE level:")
print(level_profile.to_string())

level_profile.to_csv(os.path.join(OUTPUT_DIR, 'step12_he_level_profiles.csv'))

# Radar chart by HE level
level_data = {}
for level in df_clean['HE_level'].unique():
    sub = df_clean[df_clean['HE_level']==level]
    level_data[level] = [sub[f'{s}_score'].mean() for s in SECTION_ITEMS.keys()]
radar_chart(level_data, 'HE Profiles by Harmonic Level', 'step12_radar_he_levels.png')

print("  Saved: step12_he_level_profiles.csv, step12_radar_he_levels.png")


# ════════════════════════════════════════════════════════════════════════════
# STEP 13 – EXPORT CLEAN DATASET
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 13 – Export clean dataset")
print("="*60)

export_cols = (
    ['CASE','STARTED','TIME_SUM','FINISHED','MISSING'] +
    LIKERT_COLS +
    SCORE_COLS +
    ['HE_index','Salus_score','Humanitas_score','Humus_score','HE_level'] +
    ['PI01','PI02','PI04','Age_label','Gender_label','Education_label']
)
df_export = df_clean[[c for c in export_cols if c in df_clean.columns]]
df_export.to_csv(os.path.join(OUTPUT_DIR, 'step13_clean_dataset.csv'), index=False)
print(f"  Saved: step13_clean_dataset.csv  ({len(df_export)} rows, {len(df_export.columns)} columns)")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINTOUT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ANALYSIS COMPLETE – Files generated:")
print("="*60)
files = [
    ('step3_item_descriptives.csv',     'Descriptive stats per Likert item'),
    ('step3_section_descriptives.csv',  'Descriptive stats per section'),
    ('step4_cronbach_alpha.csv',         'Reliability (Cronbach alpha) per section'),
    ('step5_demographic_analysis.csv',   'Mann-Whitney U tests by gender & age'),
    ('step6_correlation_matrix.png',     'Spearman correlation heatmap'),
    ('step6_correlation_matrix.csv',     'Correlation matrix (CSV)'),
    ('step7a_radar_overall.png',         'Radar chart – overall HE profile'),
    ('step7b_radar_gender.png',          'Radar chart – by gender'),
    ('step7c_radar_age.png',             'Radar chart – by age group'),
    ('step8_item_distributions.png',     'Response distribution per item'),
    ('step8_section_means.png',          'Section mean bar chart'),
    ('step9_response_time.png',          'Response time distribution + HE scatter'),
    ('step10_word_frequency.png',        'Top words bar chart'),
    ('step10_word_frequency.csv',        'Word frequency table'),
    ('step11_text_quality.png',          'A7 open-text quality charts'),
    ('step11_text_quality_metrics.csv',  'Per-response text quality metrics'),
    ('step12_he_level_profiles.csv',     'Profiles by HE maturity level'),
    ('step12_radar_he_levels.png',       'Radar chart – by HE level'),
    ('step13_clean_dataset.csv',         'Full clean dataset with all scores'),
]
for fname, desc in files:
    print(f"  {fname:<42} ← {desc}")

print("""
NEXT STEPS (manual):
─────────────────────────────────────────────────────────
1. Review step4_cronbach_alpha.csv — if any section has α < 0.70,
   treat that section score as exploratory only.

2. Open step13_clean_dataset.csv and manually add a GROUP column
   (1=Knowledge Learning, 2=Information Extraction, 3=Knowledge
   Discovery, 4=Interaction) if you can recover assignments.
   Then rerun with between-group Kruskal-Wallis tests (code below).

3. For the A7 Italian open-text, consider:
   a) Manual thematic coding per group hypothesis
   b) Using an AI tool to classify each response into depth levels

BONUS – Between-group test code (add when GROUP is known):
─────────────────────────────────────────────────────────
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

for score_col in SCORE_COLS + ['HE_index']:
    groups = [df_clean[df_clean['GROUP']==g][score_col].dropna() for g in [1,2,3,4]]
    stat, p = kruskal(*groups)
    print(f"{score_col}: H={stat:.2f}, p={p:.3f}")
    if p < 0.05:
        dunn = posthoc_dunn(df_clean, val_col=score_col, group_col='GROUP', p_adjust='bonferroni')
        print(dunn)
""")