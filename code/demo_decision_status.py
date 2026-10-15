"""Reproducible demonstration for Section 6 of the review.

Illustrates how the decision status of a cluster-count rule (RANGE, SUBST-D,
SUBST-C, SUBST-R) shows up in practice on small synthetic data:
  (i)  behaviour under a similarity rescaling x -> a x (a = 0.1, 1, 10),
  (ii) behaviour on structureless data (uniform square, true k = 1),
  (iii) behaviour when the true k exceeds the user-supplied k_max.
This is an illustration on toy data, not a benchmark.

Requirements: numpy, scipy, scikit-learn (>=1.3 for HDBSCAN), matplotlib.
Usage: python demo_decision_status.py --reps 20 --out results/
"""
import argparse, os, json, time
import numpy as np
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

# ----------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
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
