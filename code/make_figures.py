"""Figures for the review. Usage: python make_figures.py <project_dir>
Fig 1: family x decision-status counts from data/methods_table.csv
Fig 2: demonstration results from results/demo_khat.csv (13 rules, incl. two AUTO rules)"""
import sys, os, csv, collections, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

base = sys.argv[1]
out = os.path.join(base, 'figures'); os.makedirs(out, exist_ok=True)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.linewidth': 0.6, 'pdf.fonttype': 42})

# ---------------- Figure 1
rows = list(csv.DictReader(open(os.path.join(base, 'data/methods_table.csv'))))
FAM = {}
for r in rows: FAM[int(r['family'])] = r['family_name']
fams = [f for f in range(1, 17)] + [0]
stats_ = ['AUTO', 'AUTO-SET', 'RANGE', 'SUBST', 'NOT-AUTO', 'SUPERVISED', 'N/A', 'UNRESOLVED']
lab = {'AUTO': 'AUTO', 'AUTO-SET': 'AUTO-SET', 'RANGE': 'RANGE', 'SUBST': 'SUBST', 'NOT-AUTO': 'NOT-AUTO',
       'SUPERVISED': 'SUP', 'N/A': 'n/a', 'UNRESOLVED': 'UNRES'}
M = collections.Counter((int(r['family']), r['status']) for r in rows)
cmap = LinearSegmentedColormap.from_list('seq', ['#f4f8fd', '#2a78d6', '#0d3f80'])
fig, ax = plt.subplots(figsize=(6.7, 4.6))
vmax = max(M.values())
for i, f in enumerate(fams):
    for j, s in enumerate(stats_):
        v = M.get((f, s), 0)
        col = cmap(0.08 + 0.92 * v / vmax) if v else '#ffffff'
        ax.add_patch(plt.Rectangle((j, i), 0.94, 0.9, color=col, lw=0))
        if v:
            ax.text(j + 0.47, i + 0.45, str(v), ha='center', va='center', fontsize=7,
                    color='white' if v / vmax > 0.45 else INK)
tot_f = collections.Counter(int(r['family']) for r in rows)
tot_s = collections.Counter(r['status'] for r in rows)
for i, f in enumerate(fams):
    ax.text(len(stats_) + 0.15, i + 0.45, str(tot_f[f]), va='center', fontsize=7, color=INK2)
for j, s in enumerate(stats_):
    ax.text(j + 0.47, len(fams) + 0.2, str(tot_s[s]), ha='center', va='top', fontsize=7, color=INK2)
ax.text(len(stats_) + 0.15, len(fams) + 0.2, str(len(rows)), va='top', fontsize=7, color=INK, weight='bold')
ax.set_xlim(0, len(stats_) + 0.8); ax.set_ylim(len(fams) + 0.9, -0.1)
ax.set_xticks([j + 0.47 for j in range(len(stats_))]); ax.set_xticklabels([lab[s] for s in stats_])
ax.xaxis.tick_top()
ax.set_yticks([i + 0.45 for i in range(len(fams))])
ax.set_yticklabels([(f'F{f} ' if f else 'F-- ') + FAM[f] for f in fams])
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xlabel('Axis 2: decision status (column totals at the bottom; row totals at the right)', color=INK2)
fig.tight_layout()
fig.savefig(os.path.join(out, 'fig1_taxonomy.pdf')); fig.savefig(os.path.join(out, 'fig1_taxonomy.eps')); fig.savefig(os.path.join(out, 'fig1_taxonomy.png'), dpi=200)

# ---------------- Figure 2
d = list(csv.DictReader(open(os.path.join(base, 'results/demo_khat.csv'))))
conds = ['5 blobs, a=0.1', '5 blobs, a=1', '5 blobs, a=10', 'uniform square (k=1)', '12 blobs, k_max=10']
titles = ['5 blobs\nscale ×0.1', '5 blobs\nscale ×1', '5 blobs\nscale ×10', 'Uniform square\ntrue k = 1', '12 blobs\n$k_{\\max}$ = 10']
ktrue = [5, 5, 5, 1, 12]
meths = []
for r in d:
    if r['method'] not in meths: meths.append(r['method'])
status = {r['method']: r['status'] for r in d}
col = {'RANGE': '#52514e', 'SUBST-C': '#2a78d6', 'SUBST-D': '#eb6834', 'SUBST-R': '#1baf7a', 'AUTO': '#8e44ad'}
mk = {'RANGE': 'o', 'SUBST-C': 's', 'SUBST-D': 'D', 'SUBST-R': '^', 'AUTO': 'p'}
V = collections.defaultdict(list)
for r in d: V[(r['method'], r['condition'])].append(int(r['k_hat']))
fig, axes = plt.subplots(1, 5, figsize=(7.0, 3.8), sharey=True)
ypos = {m: i for i, m in enumerate(meths)}
for ax, c, t, kt in zip(axes, conds, titles, ktrue):
    ax.set_xscale('symlog', linthresh=1)
    ax.axvline(kt, color=INK, lw=0.8, ls=(0, (3, 2)))
    for m in meths:
        v = V[(m, c)]; s = status[m]; y = ypos[m]
        ax.plot([min(v), max(v)], [y, y], color=col[s], lw=1.6, solid_capstyle='round')
        ax.plot(st.median(v), y, mk[s], color=col[s], ms=4.5, mec='white', mew=0.6)
    ax.set_title(t, fontsize=7.5, color=INK)
    ax.set_xlim(-0.3, 600); ax.set_xticks([0, 1, 5, 20, 100])
    ax.set_xticklabels(['0', '1', '5', '20', '100'], fontsize=6)
    ax.grid(axis='x', color=GRID, lw=0.4); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
axes[0].set_yticks(range(len(meths))); axes[0].set_yticklabels(meths); axes[0].invert_yaxis()
fig.text(0.6, 0.015, 'Estimated number of clusters $\\hat k$ (median and range over 20 replicates; symmetric-log axis; dashed line = true $k$)',
         ha='center', fontsize=7, color=INK2)
from matplotlib.lines import Line2D
h = [Line2D([0], [0], marker=mk[s], color=col[s], lw=1.6, ms=4.5, mec='white', label=s) for s in col]
fig.legend(handles=h, loc='upper center', ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.6, 1.0))
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig(os.path.join(out, 'fig2_demo.pdf')); fig.savefig(os.path.join(out, 'fig2_demo.eps')); fig.savefig(os.path.join(out, 'fig2_demo.png'), dpi=200)
print('ok')
