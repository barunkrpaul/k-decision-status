"""Reproducible demonstration for Section 6 of the review.

Illustrates how the decision status of a cluster-count rule (RANGE, SUBST-D,
SUBST-C, SUBST-R, AUTO) shows up in practice on small synthetic data:
  (i)  behaviour under a similarity rescaling x -> a x (a = 0.1, 1, 10),
  (ii) behaviour on structureless data (uniform square, true k = 1),
  (iii) behaviour when the true k exceeds the user-supplied k_max.
This is an illustration on toy data, not a benchmark.

Thirteen rules are applied: four RANGE rules, three SUBST-D rules, two
SUBST-C settings of G-means, two SUBST-R settings of HDBSCAN, and two rules
coded AUTO in the review, U-k-means (Sinaga and Yang, 2020) and k*-means
(Mahon and Lapata, 2025). The AUTO rules take no candidate range and no
user-set controlling quantity; their fixed design constants are listed in
the docstrings below and in Table S1 of Online Resource 1.

Requirements: numpy, scipy, scikit-learn (>=1.3 for HDBSCAN), matplotlib.
Usage: python demo_decision_status.py --reps 20 --out results/
The script prints the versions of numpy, scipy and scikit-learn that were
used, and writes them to results/software_versions.json.
"""
import argparse, os, json, time, platform
import numpy as np
import scipy
import sklearn
from scipy import stats
from sklearn.cluster import KMeans, MeanShift, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture

N = 500
KMAX = 10

# ----------------------------------------------------------------- data
def blobs(rng, k, n=N, radius=None, sd=1.0):
    """k isotropic Gaussian blobs, centres equally spaced on a circle."""
    radius = radius if radius is not None else max(6.0, 1.6 * k)
    ang = 2 * np.pi * np.arange(k) / k
    C = np.c_[radius * np.cos(ang), radius * np.sin(ang)]
    lab = rng.integers(0, k, size=n)
    return C[lab] + sd * rng.standard_normal((n, 2))

def uniform_square(rng, n=N, side=10.0):
    return side * rng.random((n, 2))

# ------------------------------------------------------------- RANGE rules
def km(X, k, seed):
    return KMeans(n_clusters=k, n_init=5, random_state=seed).fit(X)

def silhouette_k(X, seed, kmax=KMAX):
    s = [silhouette_score(X, km(X, k, seed).labels_) for k in range(2, kmax + 1)]
    return 2 + int(np.argmax(s))

def ch_k(X, seed, kmax=KMAX):
    s = [calinski_harabasz_score(X, km(X, k, seed).labels_) for k in range(2, kmax + 1)]
    return 2 + int(np.argmax(s))

def gap_k(X, seed, kmax=KMAX, B=20):
    """Tibshirani et al. (2001), uniform reference over the bounding box, 1-s.e. rule."""
    rng = np.random.default_rng(seed)
    lo, hi = X.min(0), X.max(0)
    logW, logWref, sk = [], [], []
    for k in range(1, kmax + 1):
        logW.append(np.log(km(X, k, seed).inertia_))
        ref = [np.log(km(lo + (hi - lo) * rng.random(X.shape), k, seed).inertia_) for _ in range(B)]
        logWref.append(np.mean(ref)); sk.append(np.std(ref) * np.sqrt(1 + 1 / B))
    gap = np.array(logWref) - np.array(logW)
    for k in range(1, kmax):
        if gap[k - 1] >= gap[k] - sk[k]:
            return k
    return kmax

def bic_k(X, seed, kmax=KMAX):
    """Spherical Gaussian mixture, BIC with fitted variances (k = 1..kmax)."""
    b = [GaussianMixture(k, covariance_type='spherical', n_init=2, random_state=seed).fit(X).bic(X)
         for k in range(1, kmax + 1)]
    return 1 + int(np.argmin(b))

# ----------------------------------------------- SUBST-D rules (data units)
def dpmeans_k(X, lam, max_iter=100):
    """Kulis and Jordan (2012) batch DP-means; lam in squared data units."""
    mu = [X.mean(0)]
    z = np.zeros(len(X), int)
    for _ in range(max_iter):
        changed = False
        for i, x in enumerate(X):
            d = ((np.array(mu) - x) ** 2).sum(1)
            j = int(np.argmin(d))
            if d[j] > lam:
                mu.append(x.copy()); j = len(mu) - 1
            if z[i] != j:
                z[i] = j; changed = True
        keep = np.unique(z)
        mu = [X[z == j].mean(0) for j in keep]
        z = np.searchsorted(keep, z)
        if not changed:
            break
    return len(mu)

def meanshift_k(X, h):
    return len(MeanShift(bandwidth=h, bin_seeding=True).fit(X).cluster_centers_)

def dbscan_k(X, eps, min_samples=10):
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_
    return len(set(lab) - {-1})

# --------------------------------------- SUBST-C rule (probability scale)
def a2star(y):
    """Anderson-Darling A^2 for normality with estimated mean and variance,
    with the small-sample adjustment A*^2 = A^2 (1 + 4/n - 25/n^2)."""
    n = len(y)
    z = np.sort((y - y.mean()) / y.std(ddof=1))
    F = np.clip(stats.norm.cdf(z), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    A2 = -n - np.mean((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1])))
    return A2 * (1 + 4 / n - 25 / n ** 2)

def ad_critical(alpha, n_sim=100000, n=200, seed=12345):
    """Monte Carlo critical value of A*^2 under normality (parameters estimated)."""
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((n_sim, n))
    Y = (Y - Y.mean(1, keepdims=True)) / Y.std(1, ddof=1, keepdims=True)
    Z = np.sort(Y, 1)
    F = np.clip(stats.norm.cdf(Z), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    A2 = -n - np.mean((2 * i - 1) * (np.log(F) + np.log(1 - F[:, ::-1])), 1)
    A2 *= (1 + 4 / n - 25 / n ** 2)
    return float(np.quantile(A2, 1 - alpha))

def gmeans_k(X, crit, seed, min_size=8, max_k=60):
    """G-means (Hamerly and Elkan 2003): split while the projected AD test rejects."""
    centers = X.mean(0, keepdims=True)
    while True:
        fit = KMeans(n_clusters=len(centers), init=centers, n_init=1, random_state=seed).fit(X)
        centers, lab = fit.cluster_centers_, fit.labels_
        new = []
        for j, c in enumerate(centers):
            P = X[lab == j]
            if len(P) < min_size:
                new.append(c); continue
            ev, evec = np.linalg.eigh(np.cov(P.T))
            m = evec[:, -1] * np.sqrt(2 * ev[-1] / np.pi)
            two = KMeans(n_clusters=2, init=np.array([c + m, c - m]), n_init=1, random_state=seed).fit(P)
            v = two.cluster_centers_[0] - two.cluster_centers_[1]
            if np.allclose(v, 0):
                new.append(c); continue
            y = P @ v / (v @ v)
            if a2star(y) > crit:
                new.extend(two.cluster_centers_)
            else:
                new.append(c)
        if len(new) == len(centers) or len(new) > max_k:
            return len(new) if len(new) <= max_k else max_k
        centers = np.array(new)

# ------------------------------------------ SUBST-R rule (structural size)
def hdbscan_k(X, mcs):
    lab = HDBSCAN(min_cluster_size=mcs).fit(X).labels_
    return len(set(lab) - {-1})

# ------------------------------------------------------------- AUTO rules
def ukmeans_k(X, tol=0.1, max_iter=500, beta_zero_after=60):
    """U-k-means (Sinaga and Yang, IEEE Access 8:80716-80727, 2020).

    Starts from one cluster per point (c = n, alpha_k = 1/n) and discards
    clusters through an entropy-penalised objective. The implementation
    follows the paper and the authors' reference code:
      membership: u_ik = 1 if k = argmin_k ||x_i - a_k||^2 - beta ln alpha_k
      beta       = exp(-c / 450)                      (fixed schedule)
      alpha_k   <- (1/n) sum_i u_ik + (gamma/beta) alpha_k (ln alpha_k - sum_s alpha_s ln alpha_s)
      discard    clusters with alpha_k <= 1/n, renormalise
      gamma     <- min( (1/c) sum_k exp(-eta n |alpha_k^new - alpha_k|),
                        (1 - max_k n_k/n) / (-max_k alpha_k sum_k alpha_k ln alpha_k) )
      eta        = min(1, t^{-floor(d/2 - 1)})
      gamma      = 0 once t >= 60 and c has stopped changing
      stop       when max_k ||a_k^new - a_k|| < tol or c = 1.
    The constants 450, 60 and tol = 0.1 (in data units) are the design
    constants of the source; none is presented there as a tuning parameter.
    Returns the number of surviving clusters."""
    n, d = X.shape
    A = X.copy() + 1e-4
    alpha = np.full(n, 1.0 / n)
    c = n
    beta, gamma = 1.0, 1.0
    for t in range(1, max_iter + 1):
        D = ((X[:, None, :] - A[None, :, :]) ** 2).sum(2)          # n x c
        score = D - beta * np.log(alpha)[None, :]
        if t == 1:                                                 # exclude a point's own singleton
            score[np.arange(n), np.arange(n)] = np.inf
        z = np.argmin(score, 1)
        U = np.zeros((n, c)); U[np.arange(n), z] = 1.0
        beta = np.exp(-c / 450.0)
        share = U.sum(0) / n
        ent = np.sum(alpha * np.log(alpha))
        alpha_new = share + (gamma / beta) * alpha * (np.log(alpha) - ent)
        eta = min(1.0, (1.0 / t) ** int(np.floor(d / 2.0 - 1)))
        g1 = np.mean(np.exp(-eta * n * np.abs(alpha_new - alpha)))
        g2 = (1.0 - share.max()) / (-alpha.max() * ent) if ent < 0 else np.inf
        gamma_new = min(g1, g2)
        keep = np.where(alpha_new > 1.0 / n)[0]
        if len(keep) == 0:
            return 1
        c_new = len(keep)
        if t >= beta_zero_after and c_new == c:
            gamma_new = 0.0
        alpha = alpha_new[keep]; alpha = alpha / alpha.sum()
        U = U[:, keep]
        empty = U.sum(1) == 0                                      # points whose cluster was discarded
        if empty.any():
            z2 = np.argmin(D[np.ix_(empty, keep)], 1)
            U[np.where(empty)[0], z2] = 1.0
        A_new = (U.T @ X) / np.maximum(U.sum(0)[:, None], 1e-12)
        if c_new == c:
            err = np.max(np.sqrt(((A_new - A[keep]) ** 2).sum(1)))
        else:
            err = np.inf
        A, c, gamma = A_new, c_new, gamma_new
        if c == 1 or err < tol:
            break
    return int(c)


def _precision_bits(X):
    """Smallest number of bits that represents every coordinate of X exactly
    on a uniform grid: m = ceil(log2(range / smallest non-zero gap))."""
    v = np.unique(np.sort(X.ravel()))
    gaps = np.diff(v)
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return 1
    return int(max(1, np.ceil(np.log2((v[-1] - v[0]) / gaps.min()))))

def kstar_means_k(X, seed, patience=5, max_cycles=300, max_k=400):
    """k*-means (Mahon and Lapata, arXiv:2505.11904, 2025).

    Minimises the two-part description length of equation (1) of the
    preprint along a single split-and-merge path that starts from k = 1:
      L(P) = k d m + n log2 k + (1/2 ln 2) sum_S Q(S)   [bits],
    where m is the per-coordinate precision computed from the data and the
    residual term codes each point under a Gaussian of unit variance centred
    on its centroid. Each cluster keeps two k-means++-seeded subclusters.
    One cycle is: a k-means step on clusters and subclusters; a split of the
    cluster whose replacement by its subclusters lowers L most, if any; and
    otherwise a merge of the closest pair of centroids if that lowers L. The
    procedure stops when L has not improved for `patience` cycles. There is
    no candidate range and no significance level. The unit-variance residual
    code is the reason the rule is not invariant to rescaling.
    Returns the number of clusters at the best description length."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    m = _precision_bits(X)
    log2 = np.log(2.0)

    def seed_two(P):
        if len(P) < 2:
            return np.vstack([P[0], P[0]])
        i0 = rng.integers(len(P))
        d0 = ((P - P[i0]) ** 2).sum(1)
        if d0.sum() <= 0:
            return np.vstack([P[i0], P[i0]])
        i1 = rng.choice(len(P), p=d0 / d0.sum())
        return np.vstack([P[i0], P[i1]])

    def Q(P, c):
        return ((P - c) ** 2).sum()

    def cost(k, resid):
        return k * d * m + n * np.log2(k) + resid / (2 * log2)

    centers = X.mean(0, keepdims=True)
    subs = [seed_two(X)]
    best = np.inf; stale = 0; best_k = 1
    for cycle in range(max_cycles):
        # k-means step on clusters
        k = len(centers)
        lab = np.argmin(((X[:, None, :] - centers[None, :, :]) ** 2).sum(2), 1)
        for j in range(k):
            if np.any(lab == j):
                centers[j] = X[lab == j].mean(0)
        # k-means step on subclusters
        for j in range(k):
            P = X[lab == j]
            if len(P) == 0:
                continue
            sl = np.argmin(((P[:, None, :] - subs[j][None, :, :]) ** 2).sum(2), 1)
            for s in (0, 1):
                if np.any(sl == s):
                    subs[j][s] = P[sl == s].mean(0)
        # maybe split
        resid = sum(Q(X[lab == j], centers[j]) for j in range(k))
        deltas = []
        for j in range(k):
            P = X[lab == j]
            if len(P) < 2:
                deltas.append(np.inf); continue
            sl = np.argmin(((P[:, None, :] - subs[j][None, :, :]) ** 2).sum(2), 1)
            if not (np.any(sl == 0) and np.any(sl == 1)):
                deltas.append(np.inf); continue
            qs = Q(P[sl == 0], subs[j][0]) + Q(P[sl == 1], subs[j][1])
            dres = qs - Q(P, centers[j])
            deltas.append(cost(k + 1, resid + dres) - cost(k, resid))
        jbest = int(np.argmin(deltas))
        if deltas[jbest] < 0 and k < max_k:
            P = X[lab == jbest]
            sl = np.argmin(((P[:, None, :] - subs[jbest][None, :, :]) ** 2).sum(2), 1)
            c0, c1 = P[sl == 0].mean(0), P[sl == 1].mean(0)
            centers = np.vstack([np.delete(centers, jbest, 0), c0, c1])
            subs = [s for i, s in enumerate(subs) if i != jbest] + [seed_two(P[sl == 0]), seed_two(P[sl == 1])]
        elif k > 1:
            # maybe merge the closest pair
            lab = np.argmin(((X[:, None, :] - centers[None, :, :]) ** 2).sum(2), 1)
            Dc = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(2)
            np.fill_diagonal(Dc, np.inf)
            a, b = np.unravel_index(int(np.argmin(Dc)), Dc.shape)
            Pa, Pb = X[lab == a], X[lab == b]
            Pu = np.vstack([Pa, Pb])
            if len(Pu) > 0:
                cu = Pu.mean(0)
                dres = Q(Pu, cu) - (Q(Pa, centers[a]) if len(Pa) else 0) - (Q(Pb, centers[b]) if len(Pb) else 0)
                if cost(k - 1, resid + dres) - cost(k, resid) < 0:
                    keep = [j for j in range(k) if j not in (a, b)]
                    newsub = np.vstack([centers[a], centers[b]])
                    centers = np.vstack([centers[keep], cu])
                    subs = [subs[j] for j in keep] + [newsub]
        # evaluate
        k = len(centers)
        lab = np.argmin(((X[:, None, :] - centers[None, :, :]) ** 2).sum(2), 1)
        resid = sum(Q(X[lab == j], centers[j]) for j in range(k) if np.any(lab == j))
        L = cost(k, resid)
        if L < best - 1e-9:
            best, best_k, stale = L, k, 0
        else:
            stale += 1
        if stale >= patience:
            break
    return int(best_k)

# ----------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    versions = {'python': platform.python_version(), 'numpy': np.__version__,
                'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__}
    print('software versions:', json.dumps(versions), flush=True)
    json.dump(versions, open(os.path.join(a.out, 'software_versions.json'), 'w'), indent=1)
    t0 = time.time()
    crit = {0.05: ad_critical(0.05), 0.01: ad_critical(0.01), 1e-4: 1.8692}
    json.dump({str(k): v for k, v in crit.items()}, open(os.path.join(a.out, 'ad_critical_values.json'), 'w'), indent=1)

    conditions = [  # (label, true k, generator, scale)
        ('5 blobs, a=0.1', 5, lambda r: blobs(r, 5), 0.1),
        ('5 blobs, a=1', 5, lambda r: blobs(r, 5), 1.0),
        ('5 blobs, a=10', 5, lambda r: blobs(r, 5), 10.0),
        ('uniform square (k=1)', 1, lambda r: uniform_square(r), 1.0),
        ('12 blobs, k_max=10', 12, lambda r: blobs(r, 12), 1.0),
    ]
    methods = [
        ('Silhouette', 'RANGE', lambda X, s: silhouette_k(X, s)),
        ('Calinski-Harabasz', 'RANGE', lambda X, s: ch_k(X, s)),
        ('Gap statistic', 'RANGE', lambda X, s: gap_k(X, s)),
        ('BIC (spherical GMM)', 'RANGE', lambda X, s: bic_k(X, s)),
        ('DP-means, lambda=16', 'SUBST-D', lambda X, s: dpmeans_k(X, 16.0)),
        ('Mean shift, h=1.5', 'SUBST-D', lambda X, s: meanshift_k(X, 1.5)),
        ('DBSCAN, eps=0.8', 'SUBST-D', lambda X, s: dbscan_k(X, 0.8)),
        ('G-means, alpha=0.05', 'SUBST-C', lambda X, s: gmeans_k(X, crit[0.05], s)),
        ('G-means, alpha=1e-4', 'SUBST-C', lambda X, s: gmeans_k(X, crit[1e-4], s)),
        ('HDBSCAN, mcs=10', 'SUBST-R', lambda X, s: hdbscan_k(X, 10)),
        ('HDBSCAN, mcs=100', 'SUBST-R', lambda X, s: hdbscan_k(X, 100)),
        ('U-k-means', 'AUTO', lambda X, s: ukmeans_k(X)),
        ('k*-means', 'AUTO', lambda X, s: kstar_means_k(X, s)),
    ]
    rows = []
    for rep in range(a.reps):
        seed = 1000 + rep
        for clab, ktrue, gen, scale in conditions:
            X = gen(np.random.default_rng(seed)) * scale
            for mlab, status, f in methods:
                rows.append(dict(rep=rep, seed=seed, condition=clab, k_true=ktrue,
                                 method=mlab, status=status, k_hat=int(f(X, seed))))
        print(f'rep {rep} done, {time.time() - t0:.0f}s', flush=True)
    import csv
    with open(os.path.join(a.out, 'demo_khat.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('done', time.time() - t0)

if __name__ == '__main__':
    main()
