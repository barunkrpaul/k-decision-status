# Decision status of methods for choosing k in K-means

Data and code for the review article:

> Paul, B. K. and Giri, K. *Determining the Number of Clusters in K-means and Related Methods: A Critical Review and a Decision-Status Taxonomy, 2016–2026.* Manuscript submitted to *Artificial Intelligence Review* (Springer Nature).

The repository holds the coded corpus of 110 methods, the code of the worked demonstration (Section 6 of the article) and its raw results. These files are the same as Online Resources 2 to 4 of the article.

## Contents

| Path | What it is | Online Resource |
| --- | --- | --- |
| `data/methods_table.csv` | Coding of the 110 methods on both axes of the taxonomy | 2 |
| `data/refs.bib` | Bibliography; the `refs` column of the table uses these keys | 2 |
| `code/demo_decision_status.py` | The demonstration (five synthetic conditions, eleven rules, 20 replicates) | 3 |
| `code/make_figures.py` | Draws Fig. 1 and Fig. 2 of the article | 3 |
| `results/demo_khat.csv` | Raw results of the demonstration (1,100 rows) | 4 |
| `results/ad_critical_values.json` | Anderson–Darling critical values used by G-means | 4 |
| `figures/` | Fig. 1 and Fig. 2 as produced by `make_figures.py` | – |

## Reproducing the results

The results were produced with Python 3.13, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1 and Matplotlib 3.11.2.

```
pip install -r requirements.txt
python code/demo_decision_status.py --reps 20 --out results
python code/make_figures.py .
```

The demonstration takes about 6 minutes on one CPU core. With the versions above, it reproduces `results/demo_khat.csv` exactly. Other versions may give small differences.

### Settings of the demonstration

- Data seeds 1000 to 1019; the same seed is used for the algorithms. K-means uses five restarts.
- Range rules (Silhouette, Calinski–Harabasz, gap statistic, BIC of a spherical Gaussian mixture) evaluate k = 2, ..., 10; the gap statistic and BIC also evaluate k = 1. The gap statistic uses B = 20 uniform draws from the axis-aligned bounding box and the one-standard-error rule.
- DP-means: λ = 16. Mean shift: bandwidth 1.5 (bin seeding). DBSCAN: ε = 0.8, 10 neighbours.
- G-means: Anderson–Darling statistic with the small-sample adjustment; critical value 0.766 for α = 0.05 (Monte Carlo, 10^5 normal samples of size 200) and 1.8692 for α = 10^-4. Clusters with fewer than 8 points are not tested, and the search stops at 60 clusters.
- HDBSCAN: minimum cluster size 10 and 100 (`min_samples` left at its default, which equals the minimum cluster size).

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
| `status` | Axis 2: AUTO, AUTO-SET (shown as AUTO† in the article), RANGE, SUBST, NOT-AUTO, SUPERVISED, N/A, UNRESOLVED |
| `subst_type` | For SUBST only: C calibrated, D dimensional, P prior, R relative or structural |
| `upper_bound` | Y if a candidate set, an upper bound or an initial over-count is required |
| `can_return_k1` | Y, N or ? (not established from the sources) |
| `user_inputs` | Quantities the user must supply |
| `status_original` | Status in the original coding, before the rulebook was applied |
| `basis` | S checked against the primary source; S* checked in part; T taken from the extraction made during corpus assembly |
| `note` | Remarks |

The rulebook used to assign the status is Section S1 of Online Resource 1 of the article.

## Licence

Code and Data : MIT licence (see `LICENSE`) for all files.

## How to cite

Please cite the article. Citation details will be added here after publication; see also `CITATION.cff`.
