# Decision status of methods for choosing k in K-means

Data and code for the review article:

> Paul, B. K. and Giri, K. *How Automatic Is "Automatic" Clustering? A Critical Review and Decision-Status Taxonomy for Choosing k in K-Means and Related Methods.* Manuscript submitted to *Artificial Intelligence Review* (Springer Nature).

The repository holds the coded corpus of 110 methods, the code of the worked demonstration (Section 6 of the article) and its raw results, and the second-coding sheet used for the reliability check of the coding (Section 2.4). These files are the same as Online Resources 2 to 4 of the article. Archived copy: https://doi.org/10.5281/zenodo.23411132

## Contents

| Path | What it is | Online Resource |
| --- | --- | --- |
| `data/methods_table.csv` | Coding of the 110 methods on both axes of the taxonomy | 2 |
| `data/refs.bib` | Bibliography; the `refs` column of the table uses these keys | 2 |
| `code/demo_decision_status.py` | The demonstration (five synthetic conditions, thirteen rules, 20 replicates) | 3 |
| `code/make_figures.py` | Draws Fig. 1 and Fig. 2 of the article | 3 |
| `results/demo_khat.csv` | Raw results of the demonstration (1,300 rows) | 4 |
| `results/ad_critical_values.json` | Anderson–Darling critical values used by G-means | 4 |
| `results/software_versions.json` | Versions of Python, NumPy, SciPy and scikit-learn that produced `demo_khat.csv` | 4 |
| `figures/` | Fig. 1 and Fig. 2 as produced by `make_figures.py` (pdf, eps, png) | – |
| `coding/second_coding_sheet_blank.xlsx` | Second-coder sheet: 30 methods drawn at random (seed 2026), the rulebook, empty cells | – |
| `coding/second_coding_filled.xlsx` | The completed second coding used for the agreement reported in Section 2.4 | – |
| `coding/kappa.py` | Computes observed agreement and Cohen's kappa between the sheet and `methods_table.csv` | – |

## Reproducing the demonstration

The results were produced with Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1 and Matplotlib 3.11.2.

```
pip install -r requirements.txt
python code/demo_decision_status.py --reps 20 --out results
python code/make_figures.py .
```

The demonstration takes about 5 minutes on one CPU core and prints the package versions it runs with. With the versions above it reproduces `results/demo_khat.csv` exactly (checked on two machines). Other versions may give small differences.

### Settings of the demonstration

- Data seeds 1000 to 1019; the same seed is used for the algorithms. K-means uses five restarts.
- Range rules (Silhouette, Calinski–Harabasz, gap statistic, BIC of a spherical Gaussian mixture) evaluate k = 2, ..., 10; the gap statistic and BIC also evaluate k = 1. The gap statistic uses B = 20 uniform draws from the axis-aligned bounding box and the one-standard-error rule.
- DP-means: λ = 16. Mean shift: bandwidth 1.5 (bin seeding). DBSCAN: ε = 0.8, 10 neighbours.
- G-means: Anderson–Darling statistic with the small-sample adjustment; critical value 0.766 for α = 0.05 (Monte Carlo, 10^5 normal samples of size 200) and 1.8692 for α = 10^-4. Clusters with fewer than 8 points are not tested, and the search stops at 60 clusters.
- HDBSCAN: minimum cluster size 10 and 100 (`min_samples` left at its default, which equals the minimum cluster size).
- U-k-means (Sinaga and Yang, 2020): implemented from the paper and the authors' MATLAB reference code; starts from one cluster per point; penalty schedule β = exp(−c/450), learning parameter set to zero after 60 iterations once c stops changing, clusters with α_k ≤ 1/n discarded, convergence tolerance 0.1 on the centre displacement. The implementation returns 3 clusters on Iris, as the authors report.
- k*-means (Mahon and Lapata, 2025): minimises the description length of equation (1) of the preprint, computed in bits with the unit-variance residual term and a per-coordinate precision taken from the data; one split-and-merge path from k = 1 with k-means++-seeded subclusters; stops after 5 cycles without improvement. The text and the pseudocode of the preprint differ in the constants of the split and merge tests, so the implementation accepts or rejects each move by the cost function itself.

## Columns of `methods_table.csv`

| Column | Meaning |
| --- | --- |
| `row` | Row number (M001 to M110 in the second-coding sheet) |
| `method` | Name of the method |
| `year` | Year of the main source |
| `refs` | BibTeX keys of the sources (see `data/refs.bib`) |
| `family`, `family_name` | Axis 1: mechanism family F1 to F16; 0 means k is an input |
| `setting` | batch, stream or deep |
| `decision_rule` | Short description of how k is decided |
| `status` | Axis 2: AUTO, AUTO-SET, RANGE, SUBST, NOT-AUTO, SUPERVISED, N/A, UNRESOLVED |
| `subst_type` | For SUBST only: C calibrated, D dimensional, P prior, R relative or structural |
| `upper_bound` | Y if a candidate set, an upper bound or an initial over-count is required |
| `can_return_k1` | Y, N or ? (not established from the sources) |
| `user_inputs` | Quantities the user must supply |
| `note` | Remarks |
| `coder_disagreement` | For the 30 rows of the reliability sample: Y if the two codings disagreed before discussion, N if they agreed; blank for rows not in the sample |

The rulebook used to assign the status is Section S1 of Online Resource 1 of the article.

## Reliability check of the coding

A random sample of 30 rows (seed 2026) was coded a second time, independently of the first coding, from the source papers and the rulebook only. Agreement with the first coding was 26 of 30 rows (86.7 per cent), Cohen's kappa = 0.80. To recompute:

```
python coding/kappa.py coding/second_coding_filled.xlsx data/methods_table.csv
```

## Licence

Code: MIT licence (see `LICENSE`). Data and results: Creative Commons Attribution 4.0 (CC BY 4.0).

## How to cite

Please cite the article. Citation details will be added here after publication; see also `CITATION.cff`.
