

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import os

OUTPUT_DIR = '.'

GROUP_LABELS = {
    1: 'Knowledge Learning\n(Textbook)',
    2: 'Information Extraction\n(NotebookLM)',
    3: 'Knowledge Discovery\n(Web Search)',
    4: 'Interaction\n(LLM)',
}
GROUP_LABELS_SHORT = {
    1: 'G1: Textbook',
    2: 'G2: NotebookLM',
    3: 'G3: Web Search',
    4: 'G4: LLM',
}
GROUP_COLORS = {
    1: '#534AB7',   # purple
    2: '#0F6E56',   # teal
    3: '#993C1D',   # coral
    4: '#185FA5',   # blue
}

SECTION_ITEMS = {
    'A0': ['A001', 'A002', 'A003'],
    'A1': ['A101', 'A102', 'A103'],
    'A2': ['A201', 'A202', 'A203'],
    'A3': ['A301', 'A302', 'A303'],
    'A4': ['A401', 'A402', 'A403'],
    'A5': ['A501', 'A502', 'A503'],
    'A6': ['A601', 'A602'],
}
SECTION_FULL_LABELS = {
    'A0': 'Entrepreneurial\nMotivation',
    'A1': 'Personal\nWell-being',
    'A2': 'Ethics &\nResponsibility',
    'A3': 'Relationships &\nStakeholders',
    'A4': 'Innovation &\nFuture',
    'A5': 'Social Function\nof Business',
    'A6': 'Self-\nAwareness',
}
LIKERT_COLS = [item for items in SECTION_ITEMS.values() for item in items]
SCORE_COLS  = [f'{s}_score' for s in SECTION_ITEMS]

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 – LOAD & MERGE
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('PART 1 – Loading data and merging group assignments')
print('='*65)

# Load survey data
df_raw = pd.read_excel('data_unical_2026-05-12_15-12.xlsx', header=None)
df = df_raw.iloc[2:].copy()
df.columns = df_raw.iloc[0].tolist()
df = df.reset_index(drop=True)
df = df[df['FINISHED'] == 1].copy().reset_index(drop=True)

for col in LIKERT_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['TIME_SUM'] = pd.to_numeric(df['TIME_SUM'], errors='coerce')
df['PI01'] = pd.to_numeric(df['PI01'], errors='coerce')
df['PI02'] = pd.to_numeric(df['PI02'], errors='coerce')

# Section scores
for sec, items in SECTION_ITEMS.items():
    df[f'{sec}_score'] = df[items].mean(axis=1)
df['HE_index'] = df[LIKERT_COLS].mean(axis=1)

# Load classification results
clf = pd.read_csv('step7_final_classifications.csv')
clf['CASE'] = clf['CASE'].astype(int)
df['CASE']  = df['CASE'].astype(int)

df = df.merge(
    clf[['CASE', 'Group', 'Confidence', 'Method', 'Rule_Applied',
         'word_count', 'url_count', 'cited_refs', 'fonti_refs',
         'he_keywords', 'trivial_answers', 'is_english', 'report_format',
         'copy_paste', 'Manual_Notes', 'Group_Label']],
    on='CASE', how='inner'
)

df['Group_Label_Short'] = df['Group'].map(GROUP_LABELS_SHORT)

# Working dataset: complete Likert + group
dg = df.dropna(subset=LIKERT_COLS).copy().reset_index(drop=True)

print(f'\n  Total merged records : {len(dg)}')
print(f'  Groups assigned      : {sorted(dg["Group"].unique())}')
print('\n  Group sizes (from methodology script):')
for g in [1, 2, 3, 4]:
    sub = dg[dg['Group'] == g]
    nh  = (sub['Confidence'] == 'HIGH').sum()
    nm  = (sub['Confidence'] == 'MEDIUM').sum()
    print(f'    G{g} {GROUP_LABELS_SHORT[g]:<22}: '
          f'n={len(sub):2d}  (HIGH={nh}, MEDIUM={nm})')

print('\n  Classification method breakdown:')
print(dg['Method'].value_counts().to_string())

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 – GROUP PROFILE TABLE
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('PART 2 – Group profile: section scores + text metrics')
print('='*65)

profile_cols = SCORE_COLS + ['HE_index', 'word_count', 'he_keywords',
                              'url_count', 'cited_refs', 'fonti_refs']
profile = dg.groupby('Group')[profile_cols].mean().round(2)
profile.index = [GROUP_LABELS_SHORT[g] for g in profile.index]
print('\n  Mean values per group:')
print(profile.to_string())

profile.to_csv(os.path.join(OUTPUT_DIR, 'part2_group_profiles.csv'))
print('\n  Saved: part2_group_profiles.csv')

# ══════════════════════════════════════════════════════════════════════════════
# PART 3 – KRUSKAL-WALLIS TESTS (LIKERT SECTIONS)
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('PART 3 – Kruskal-Wallis tests across groups (Likert sections)')
print('='*65)
print('\n  H0: No difference in section scores across the 4 learning groups')
print('  Test: Kruskal-Wallis (non-parametric, ordinal data, unequal n)\n')

kw_results = []
for sc in SCORE_COLS + ['HE_index']:
    grp_vals = [dg[dg['Group'] == g][sc].dropna().values for g in [1, 2, 3, 4]]
    stat, p  = stats.kruskal(*grp_vals)
    sig      = '**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.')
    means    = {g: dg[dg['Group'] == g][sc].mean() for g in [1, 2, 3, 4]}
    print(f'  {sc:<12}: H={stat:5.2f}  p={p:.3f}  {sig}   '
          f'G1={means[1]:.2f}  G2={means[2]:.2f}  '
          f'G3={means[3]:.2f}  G4={means[4]:.2f}')
    kw_results.append({
        'Score': sc, 'H_statistic': round(stat, 2), 'p_value': round(p, 3),
        'Significance': sig,
        **{f'G{g}_mean': round(means[g], 2) for g in [1, 2, 3, 4]}
    })

kw_df = pd.DataFrame(kw_results)
kw_df.to_csv(os.path.join(OUTPUT_DIR, 'part3_kruskal_wallis.csv'), index=False)
print('\n  Saved: part3_kruskal_wallis.csv')
print('\n  (* p<0.05   ** p<0.01   n.s. = not significant)')

# ══════════════════════════════════════════════════════════════════════════════
# PART 4 – PAIRWISE POST-HOC (BONFERRONI-CORRECTED MANN-WHITNEY)
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('PART 4 – Pairwise Mann-Whitney post-hoc (Bonferroni corrected)')
print('='*65)
print('\n  Applied to ALL section scores (not just significant ones)')
print('  Bonferroni correction: n_comparisons = 6 pairs × 8 scores\n')

pairs = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
n_comp_per_score = len(pairs)

posthoc_rows = []
for sc in SCORE_COLS + ['HE_index']:
    print(f'\n  --- {sc} ---')
    for g1, g2 in pairs:
        v1 = dg[dg['Group'] == g1][sc].dropna()
        v2 = dg[dg['Group'] == g2][sc].dropna()
        if len(v1) < 2 or len(v2) < 2:
            continue
        u_stat, p_raw = stats.mannwhitneyu(v1, v2, alternative='two-sided')
        p_adj = min(p_raw * n_comp_per_score, 1.0)
        sig   = '**' if p_adj < 0.01 else ('*' if p_adj < 0.05 else 'n.s.')
        lbl1  = GROUP_LABELS_SHORT[g1]
        lbl2  = GROUP_LABELS_SHORT[g2]
        print(f'    {lbl1} vs {lbl2:<22}: '
              f'U={u_stat:5.0f}  p_raw={p_raw:.3f}  '
              f'p_adj={p_adj:.3f}  {sig}  '
              f'[{v1.mean():.2f} vs {v2.mean():.2f}]')
        posthoc_rows.append({
            'Score': sc, 'G_A': g1, 'G_B': g2,
            'Label_A': lbl1, 'Label_B': lbl2,
            'U_statistic': round(u_stat, 1),
            'p_raw': round(p_raw, 3),
            'p_bonferroni': round(p_adj, 3),
            'Significance': sig,
            'Mean_A': round(v1.mean(), 2),
            'Mean_B': round(v2.mean(), 2),
        })

posthoc_df = pd.DataFrame(posthoc_rows)
posthoc_df.to_csv(os.path.join(OUTPUT_DIR, 'part4_posthoc_pairwise.csv'), index=False)
print('\n  Saved: part4_posthoc_pairwise.csv')

# ══════════════════════════════════════════════════════════════════════════════
# PART 5 – TEXT QUALITY COMPARISON ACROSS GROUPS
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('PART 5 – Text quality signals by group (Kruskal-Wallis)')
print('='*65)

text_signals = ['word_count', 'he_keywords', 'url_count',
                'cited_refs', 'fonti_refs', 'trivial_answers']
text_labels  = ['Word count', 'HE keywords', 'URL count',
                '[n] citations', 'Fonti refs', 'Trivial answers']

print('\n  Mean signal values per group:')
sig_means = dg.groupby('Group')[text_signals].mean().round(2)
sig_means.index = [GROUP_LABELS_SHORT[g] for g in sig_means.index]
print(sig_means.to_string())

print('\n  Kruskal-Wallis tests on text signals:')
text_kw_rows = []
for sig, lbl in zip(text_signals, text_labels):
    grp_vals = [dg[dg['Group'] == g][sig].dropna().values for g in [1, 2, 3, 4]]
    stat, p  = stats.kruskal(*grp_vals)
    significance = '**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.')
    print(f'  {lbl:<20}: H={stat:6.2f}  p={p:.4f}  {significance}')
    text_kw_rows.append({'Signal': lbl, 'H': round(stat, 2),
                         'p': round(p, 4), 'Sig': significance})

pd.DataFrame(text_kw_rows).to_csv(
    os.path.join(OUTPUT_DIR, 'part5_text_signal_tests.csv'), index=False)
print('\n  Saved: part5_text_signal_tests.csv')

# ══════════════════════════════════════════════════════════════════════════════
# PART 6 – RADAR CHART: HE PROFILES BY GROUP
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('PART 6 – Radar chart: HE profiles by group')
print('='*65)

sections = list(SECTION_ITEMS.keys())
labels   = [SECTION_FULL_LABELS[s] for s in sections]
N        = len(labels)
angles   = [n / float(N) * 2 * np.pi for n in range(N)]
angles  += angles[:1]

fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                          subplot_kw=dict(polar=True))

# Left: all groups overlaid
ax = axes[0]
for g in [1, 2, 3, 4]:
    sub    = dg[dg['Group'] == g]
    vals   = [sub[f'{s}_score'].mean() for s in sections]
    vals  += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=2.2,
            label=GROUP_LABELS_SHORT[g], color=GROUP_COLORS[g])
    ax.fill(angles, vals, alpha=0.07, color=GROUP_COLORS[g])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, size=8)
ax.set_ylim(1, 4)
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(['1', '2', '3', '4'], size=7)
ax.set_title('HE Profiles – All Groups Overlaid\n(1=least harmonic, 4=fully harmonic)',
             size=11, pad=16)
ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=9)

# Right: high-confidence only
ax2 = axes[1]
for g in [1, 2, 3, 4]:
    sub   = dg[(dg['Group'] == g) & (dg['Confidence'] == 'HIGH')]
    if len(sub) < 2:
        continue
    vals  = [sub[f'{s}_score'].mean() for s in sections]
    vals += vals[:1]
    ax2.plot(angles, vals, 'o-', linewidth=2.2,
             label=f'{GROUP_LABELS_SHORT[g]} (n={len(sub)} HIGH)',
             color=GROUP_COLORS[g])
    ax2.fill(angles, vals, alpha=0.07, color=GROUP_COLORS[g])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(labels, size=8)
ax2.set_ylim(1, 4)
ax2.set_yticks([1, 2, 3, 4])
ax2.set_yticklabels(['1', '2', '3', '4'], size=7)
ax2.set_title('HE Profiles – HIGH Confidence Cases Only',
              size=11, pad=16)
ax2.legend(loc='upper right', bbox_to_anchor=(1.55, 1.15), fontsize=9)

plt.suptitle('Harmonic Entrepreneur Profile by Learning Method Group',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'part6_radar_by_group.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: part6_radar_by_group.png')

# ══════════════════════════════════════════════════════════════════════════════
# PART 7 – BOX PLOTS: HE INDEX + EACH SECTION BY GROUP
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('PART 7 – Box plots: HE index and section scores by group')
print('='*65)

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

plot_scores = ['HE_index'] + SCORE_COLS
plot_titles = ['HE Composite Index'] + \
              [SECTION_FULL_LABELS[s].replace('\n', ' ')
               for s in SECTION_ITEMS.keys()]

for idx, (sc, title) in enumerate(zip(plot_scores, plot_titles)):
    ax = axes[idx]
    grp_data = [dg[dg['Group'] == g][sc].dropna().values for g in [1, 2, 3, 4]]

    bp = ax.boxplot(grp_data, patch_artist=True,
                    notch=False, widths=0.55,
                    medianprops=dict(color='white', linewidth=2))
    for patch, g in zip(bp['boxes'], [1, 2, 3, 4]):
        patch.set_facecolor(GROUP_COLORS[g])
        patch.set_alpha(0.80)
    for whisker in bp['whiskers']:
        whisker.set_color('#666')
    for cap in bp['caps']:
        cap.set_color('#666')
    for flier in bp['fliers']:
        flier.set_marker('o')
        flier.set_markerfacecolor('#aaa')
        flier.set_markersize(4)

    # Add mean dots
    for i, (vals, g) in enumerate(zip(grp_data, [1, 2, 3, 4])):
        if len(vals) > 0:
            ax.scatter(i + 1, np.mean(vals), color='white',
                       s=40, zorder=5, edgecolors=GROUP_COLORS[g], linewidth=1.5)

    # Kruskal-Wallis p-value annotation
    stat, p = stats.kruskal(*grp_data)
    sig = '**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.')
    ax.text(0.97, 0.97, f'K-W {sig}\np={p:.3f}',
            transform=ax.transAxes, fontsize=8,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#ccc', alpha=0.8))

    ax.set_xticklabels([f'G{g}' for g in [1, 2, 3, 4]], fontsize=9)
    ax.set_ylim(0.8, 4.2)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylabel('Score (1–4)', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.axhline(2.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

legend_patches = [
    mpatches.Patch(color=GROUP_COLORS[g],
                   label=GROUP_LABELS_SHORT[g])
    for g in [1, 2, 3, 4]
]
fig.legend(handles=legend_patches, loc='lower center',
           ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Section Score Distributions by Learning Method Group\n'
             '(○ = group mean  |  box = IQR  |  whiskers = range)',
             fontsize=13)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(OUTPUT_DIR, 'part7_boxplots_by_group.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: part7_boxplots_by_group.png')

# ══════════════════════════════════════════════════════════════════════════════
# PART 8 – TEXT SIGNAL FINGERPRINT CHART
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('PART 8 – Text signal fingerprint chart by group')
print('='*65)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

fingerprint_signals = [
    ('word_count',       'Total Word Count\n(response length)'),
    ('he_keywords',      'HE Chapter Keywords\n(Harmonic Entrepreneur terms)'),
    ('url_count',        'URL Count\n(http:// links in answers)'),
    ('cited_refs',       '[n] Inline Citations\n(NotebookLM style)'),
    ('fonti_refs',       '"Le Fonti" References\n(NotebookLM source phrases)'),
    ('trivial_answers',  'Trivial Answers\n(number or "." = gave up)'),
]

for idx, (sig, title) in enumerate(fingerprint_signals):
    ax = axes[idx]
    vals   = [dg[dg['Group'] == g][sig].mean() for g in [1, 2, 3, 4]]
    colors = [GROUP_COLORS[g] for g in [1, 2, 3, 4]]
    labels = [f'G{g}' for g in [1, 2, 3, 4]]

    bars = ax.bar(labels, vals, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.8)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.set_ylabel('Mean value', fontsize=9)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10,
                    fontweight='bold')

    # Individual data points (jittered)
    for i, g in enumerate([1, 2, 3, 4]):
        y_vals = dg[dg['Group'] == g][sig].dropna().values
        x_jit  = np.random.uniform(-0.18, 0.18, size=len(y_vals))
        ax.scatter(np.full(len(y_vals), i) + x_jit, y_vals,
                   color='white', s=20, alpha=0.7, zorder=3,
                   edgecolors=GROUP_COLORS[g], linewidth=0.8)

    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels([GROUP_LABELS_SHORT[g].split(': ')[1]
                        for g in [1, 2, 3, 4]], fontsize=8)

fig.suptitle(
    'Text Signal "Fingerprints" by Learning Group\n'
    'How each tool leaves a distinct trace in student answers',
    fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'part8_text_fingerprints.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: part8_text_fingerprints.png')

# ══════════════════════════════════════════════════════════════════════════════
# PART 9 – CONFIDENCE BREAKDOWN CHART
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('PART 9 – Classification confidence breakdown')
print('='*65)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Stacked bar: HIGH vs MEDIUM per group
high_counts = [
    (dg['Group'] == g).sum() for g in [1, 2, 3, 4]
]
# Break down by confidence
conf_data = {}
for g in [1, 2, 3, 4]:
    sub = dg[dg['Group'] == g]
    conf_data[g] = {
        'HIGH':   (sub['Confidence'] == 'HIGH').sum(),
        'MEDIUM': (sub['Confidence'] == 'MEDIUM').sum(),
    }

x       = np.arange(4)
high_v  = [conf_data[g]['HIGH']   for g in [1, 2, 3, 4]]
med_v   = [conf_data[g]['MEDIUM'] for g in [1, 2, 3, 4]]
total_v = [h + m for h, m in zip(high_v, med_v)]

bars_h = axes[0].bar(x, high_v,
                      color=[GROUP_COLORS[g] for g in [1, 2, 3, 4]],
                      alpha=0.9, label='HIGH confidence')
bars_m = axes[0].bar(x, med_v, bottom=high_v,
                      color=[GROUP_COLORS[g] for g in [1, 2, 3, 4]],
                      alpha=0.40, hatch='//', label='MEDIUM confidence')

for i, (h, m, t) in enumerate(zip(high_v, med_v, total_v)):
    if h > 0:
        axes[0].text(i, h / 2, str(h), ha='center', va='center',
                     fontsize=11, fontweight='bold', color='white')
    if m > 0:
        axes[0].text(i, h + m / 2, str(m), ha='center', va='center',
                     fontsize=11, fontweight='bold', color='#333')
    axes[0].text(i, t + 0.15, f'n={t}', ha='center', va='bottom',
                 fontsize=10)

axes[0].set_xticks(x)
axes[0].set_xticklabels([f'G{g}\n{GROUP_LABELS_SHORT[g].split(": ")[1]}'
                          for g in [1, 2, 3, 4]], fontsize=9)
axes[0].set_ylabel('Number of students', fontsize=11)
axes[0].set_title('Students per Group\n(by classification confidence)',
                  fontsize=11)
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, max(total_v) + 2)

# Method breakdown: Hard rule vs Soft score
hard_counts = dg[dg['Method'] == 'Hard rule']['Group'].value_counts().sort_index()
soft_counts = dg[dg['Method'] == 'Soft score']['Group'].value_counts().sort_index()
hard_v2 = [hard_counts.get(g, 0) for g in [1, 2, 3, 4]]
soft_v2 = [soft_counts.get(g, 0) for g in [1, 2, 3, 4]]

axes[1].bar(x - 0.2, hard_v2, width=0.38,
            color=[GROUP_COLORS[g] for g in [1, 2, 3, 4]],
            alpha=0.9, label='Hard rule (unambiguous)')
axes[1].bar(x + 0.2, soft_v2, width=0.38,
            color=[GROUP_COLORS[g] for g in [1, 2, 3, 4]],
            alpha=0.45, hatch='//', label='Soft score (weighted)')

for i, (h, s) in enumerate(zip(hard_v2, soft_v2)):
    if h > 0:
        axes[1].text(i - 0.2, h + 0.1, str(h),
                     ha='center', fontsize=10, fontweight='bold')
    if s > 0:
        axes[1].text(i + 0.2, s + 0.1, str(s),
                     ha='center', fontsize=10, fontweight='bold')

axes[1].set_xticks(x)
axes[1].set_xticklabels([f'G{g}\n{GROUP_LABELS_SHORT[g].split(": ")[1]}'
                          for g in [1, 2, 3, 4]], fontsize=9)
axes[1].set_ylabel('Number of students', fontsize=11)
axes[1].set_title('Classification Method Used\n(hard rule vs weighted soft scoring)',
                  fontsize=11)
axes[1].legend(fontsize=9)

fig.suptitle('Classification Confidence & Method Breakdown', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'part9_confidence_breakdown.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: part9_confidence_breakdown.png')

# ══════════════════════════════════════════════════════════════════════════════
# PART 10 – RESPONSE TIME BY GROUP
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('PART 10 – Response time (TIME_SUM) by group')
print('='*65)

dg_time = dg[dg['TIME_SUM'] > 60].copy()
print(f'\n  Valid TIME_SUM records (>60s): {len(dg_time)}')

print('\n  Mean completion time by group (seconds):')
for g in [1, 2, 3, 4]:
    sub = dg_time[dg_time['Group'] == g]['TIME_SUM'].dropna()
    if len(sub) > 0:
        print(f'    G{g}: mean={sub.mean():.0f}s '
              f'({sub.mean()/60:.1f} min)  '
              f'median={sub.median():.0f}s  n={len(sub)}')

grp_time = [dg_time[dg_time['Group'] == g]['TIME_SUM'].dropna().values
            for g in [1, 2, 3, 4]]
grp_time_f = [v for v in grp_time if len(v) >= 2]
if len(grp_time_f) >= 2:
    stat, p = stats.kruskal(*grp_time_f)
    sig     = '**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.')
    print(f'\n  Kruskal-Wallis TIME_SUM: H={stat:.2f}  p={p:.3f}  {sig}')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Box plot of TIME_SUM by group
bp = axes[0].boxplot(
    [dg_time[dg_time['Group'] == g]['TIME_SUM'].dropna().values / 60
     for g in [1, 2, 3, 4]],
    patch_artist=True, widths=0.55,
    medianprops=dict(color='white', linewidth=2)
)
for patch, g in zip(bp['boxes'], [1, 2, 3, 4]):
    patch.set_facecolor(GROUP_COLORS[g])
    patch.set_alpha(0.80)
axes[0].set_xticklabels(
    [GROUP_LABELS_SHORT[g].split(': ')[1] for g in [1, 2, 3, 4]],
    fontsize=9)
axes[0].set_ylabel('Completion time (minutes)', fontsize=11)
axes[0].set_title('Survey Completion Time by Group', fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

# Scatter: TIME_SUM vs HE_index coloured by group
for g in [1, 2, 3, 4]:
    sub = dg_time[dg_time['Group'] == g]
    axes[1].scatter(sub['TIME_SUM'] / 60, sub['HE_index'],
                    color=GROUP_COLORS[g], s=70, alpha=0.75,
                    edgecolors='white', linewidth=0.5,
                    label=GROUP_LABELS_SHORT[g])

# Overall trend line
all_t = dg_time['TIME_SUM'].dropna() / 60
all_h = dg_time.loc[dg_time['TIME_SUM'].notna(), 'HE_index']
if len(all_t) == len(all_h):
    m, b = np.polyfit(all_t, all_h, 1)
    x_l  = np.linspace(all_t.min(), all_t.max(), 100)
    r, pv = stats.spearmanr(all_t, all_h)
    axes[1].plot(x_l, m * x_l + b, '--', color='#888', linewidth=1.5,
                 label=f'Trend  ρ={r:.2f}  p={pv:.3f}')

axes[1].set_xlabel('Completion time (minutes)', fontsize=11)
axes[1].set_ylabel('HE Composite Index', fontsize=11)
axes[1].set_title('Time vs HE Index by Group', fontsize=11)
axes[1].legend(fontsize=8, loc='lower right')

fig.suptitle('Response Time Analysis', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'part10_response_time.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: part10_response_time.png')

# ══════════════════════════════════════════════════════════════════════════════
# PART 11 – ITEM-LEVEL HEATMAP: MEAN SCORE PER ITEM PER GROUP
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('PART 11 – Item-level heatmap: mean score per item × group')
print('='*65)

item_group_means = pd.DataFrame({
    GROUP_LABELS_SHORT[g]: dg[dg['Group'] == g][LIKERT_COLS].mean()
    for g in [1, 2, 3, 4]
})

# Row annotations: which section each item belongs to
section_of = {}
for sec, items in SECTION_ITEMS.items():
    for item in items:
        section_of[item] = SECTION_FULL_LABELS[sec].replace('\n', ' ')

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(
    item_group_means,
    annot=True, fmt='.2f',
    cmap='RdYlGn', vmin=1, vmax=4, center=2.5,
    linewidths=0.5, linecolor='white',
    ax=ax,
    cbar_kws={'label': 'Mean score (1=least harmonic, 4=fully harmonic)',
              'shrink': 0.6}
)

# Section separator lines
section_breaks = []
current_sec = None
for i, item in enumerate(LIKERT_COLS):
    sec = section_of[item]
    if sec != current_sec:
        if current_sec is not None:
            section_breaks.append(i)
        current_sec = sec
for brk in section_breaks:
    ax.axhline(brk, color='#222', linewidth=1.5)

ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
ax.set_title('Mean Likert Score per Item × Learning Group\n'
             '(horizontal lines = section boundaries)',
             fontsize=12, pad=12)
ax.set_xlabel('')
ax.set_ylabel('Survey Item', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'part11_item_heatmap.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: part11_item_heatmap.png')

item_group_means.to_csv(os.path.join(OUTPUT_DIR, 'part11_item_means_by_group.csv'))
print('  Saved: part11_item_means_by_group.csv')

# ══════════════════════════════════════════════════════════════════════════════
# PART 12 – EXPORT FULL LABELLED DATASET
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('PART 12 – Exporting full labelled dataset')
print('='*65)

export_cols = (
    ['CASE', 'Group', 'Group_Label', 'Group_Label_Short',
     'Confidence', 'Method', 'Rule_Applied'] +
    LIKERT_COLS + SCORE_COLS + ['HE_index'] +
    ['TIME_SUM', 'PI01', 'PI02', 'PI04'] +
    ['word_count', 'url_count', 'cited_refs', 'fonti_refs',
     'he_keywords', 'trivial_answers', 'is_english',
     'report_format', 'copy_paste', 'Manual_Notes']
)
df_export = dg[[c for c in export_cols if c in dg.columns]].copy()
df_export.to_csv(os.path.join(OUTPUT_DIR, 'part12_labelled_dataset.csv'),
                 index=False)
print(f'\n  Saved: part12_labelled_dataset.csv  '
      f'({len(df_export)} rows, {len(df_export.columns)} columns)')

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY PRINTOUT
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('ANALYSIS COMPLETE – Output files generated')
print('='*65)
outputs = [
    ('part2_group_profiles.csv',       'Group mean profiles (sections + text signals)'),
    ('part3_kruskal_wallis.csv',        'Kruskal-Wallis test results per section'),
    ('part4_posthoc_pairwise.csv',      'Bonferroni pairwise Mann-Whitney results'),
    ('part5_text_signal_tests.csv',     'Kruskal-Wallis on text quality signals'),
    ('part6_radar_by_group.png',        'Radar: HE profile per group (all + HIGH only)'),
    ('part7_boxplots_by_group.png',     'Box plots: all sections + HE index by group'),
    ('part8_text_fingerprints.png',     'Text signal fingerprint bars by group'),
    ('part9_confidence_breakdown.png',  'Confidence & method breakdown chart'),
    ('part10_response_time.png',        'Completion time box + scatter by group'),
    ('part11_item_heatmap.png',         'Heatmap: item-level mean score × group'),
    ('part11_item_means_by_group.csv',  'Item means by group (CSV)'),
    ('part12_labelled_dataset.csv',     'Full dataset with all scores + group labels'),
]
for fname, desc in outputs:
    print(f'  {fname:<44} ← {desc}')

print("""
KEY RESULTS SUMMARY
──────────────────────────────────────────────────────────────
See part3_kruskal_wallis.csv for significance results.
Check part4_posthoc_pairwise.csv for which group PAIRS differ.

Group sizes from the methodology script:
  G1 Knowledge Learning (Textbook)    : n=14
  G2 Information Extraction (NotebookLM): n=13
  G3 Knowledge Discovery (Web Search) : n=5
  G4 Interaction (LLM)               : n=2

NOTE: G3 (n=5) and G4 (n=2) have small samples — treat their
statistical tests as exploratory only. Bonferroni correction
is applied to control for multiple comparisons.
""")