"""
FINAL EXPERIMENT -- stripe artefact detection
=============================================

This is the run that produces the Results section.

It fixes the bug that invalidated the earlier control test: the old
control produced only two distinct values, so "take the top N" was
breaking ties by array order rather than by merit.

Four safeguards are built in:

  1. SELF-TESTS run first. If any fails, the script stops and runs
     nothing. They check that each criterion finds a planted spike,
     that none is degenerate, that all-tied input scores at chance,
     and that unrelated truth scores at chance.

  2. Ties broken at RANDOM, never by array order.

  3. Degeneracy reported for every criterion on the real data.

  4. Three independent controls, all continuous:
        C1 bound    distance from the middle of the neighbourhood range
        C2 mean     deviation from the local mean
        C3 zscore   deviation from the local mean over local std
     If any control matches hyper, the topology adds nothing and the
     paper must say so.

Truth EXCLUDES local extrema, so no criterion can win by detecting
extrema alone.

Outputs:
    stripe_raw.csv
    stripe_summary.txt
    fig_stripe.png

Runtime: about 2 minutes per volume. 30 volumes is roughly an hour.
Set N_VOLUMES = 3 for a trial run first.

Run:  python final_stripe_experiment.py
"""

import os, sys, time, csv, collections
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.stats import wilcoxon, spearmanr

# =====================================================================
# FIXED PROTOCOL -- do not change after seeing output
# =====================================================================
DATA_DIR  = r"brain_data\Task01_BrainTumour\imagesTr"
N_VOLUMES = 30
MODALITY  = 0
SEED      = 42
AMPS      = [0.2, 0.5, 1.0, 2.0]

# =====================================================================
# Criteria
# =====================================================================
def crit_median(I):
    """Classical switching-median detector. Prior art."""
    return np.abs(I - ndimage.median_filter(I, 3))

def crit_hyper(I):
    """Xi: hyperconnectedness violation. Ranks exclude the voxel itself."""
    r1 = ndimage.rank_filter(I, 1, size=3)
    r2 = ndimage.rank_filter(I, -2, size=3)
    return np.maximum(I - r2, r1 - I)

def ctrl_bound(I):
    """C1: distance from the middle of the neighbourhood range."""
    lo = ndimage.minimum_filter(I, 3); hi = ndimage.maximum_filter(I, 3)
    return np.abs(I - 0.5*(lo + hi))

def ctrl_mean(I):
    """C2: deviation from the local mean. No ranks, no topology."""
    return np.abs(I - ndimage.uniform_filter(I, 3))

def ctrl_zscore(I):
    """C3: local z-score. Standard statistical outlier detection."""
    mu = ndimage.uniform_filter(I, 5)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(I**2, 5) - mu**2, 0))
    return np.abs(I - mu) / (sd + 1e-6)

CRIT = {"median": crit_median, "hyper": crit_hyper,
        "C1_bound": ctrl_bound, "C2_mean": ctrl_mean, "C3_zscore": ctrl_zscore}
ORDER = ["median", "hyper", "C1_bound", "C2_mean", "C3_zscore"]

# =====================================================================
def rank_score(stat, truth, mask, rng):
    """Threshold-free ranking with random tie-breaking."""
    s = stat[mask].astype(np.float64)
    tf = truth[mask]
    N = int(tf.sum())
    if N == 0:
        return float("nan")
    jit = rng.random(s.shape) * 1e-12
    return float(tf[np.lexsort((jit, -s))[:N]].sum() / N)

def n_distinct(stat, mask):
    return int(len(np.unique(np.round(stat[mask].astype(np.float64), 8))))

# =====================================================================
# SELF-TESTS -- the script refuses to run if these fail
# =====================================================================
def self_test():
    print("SELF-TESTS")
    rng = np.random.default_rng(0)
    ok = True

    V = ndimage.gaussian_filter(rng.normal(0,1,(30,30,30)).astype(np.float32), 2)
    V[15,15,15] = V.max() + 50
    m = np.ones(V.shape, bool)
    t = np.zeros(V.shape, bool); t[15,15,15] = True
    for cn in ORDER:
        s = rank_score(CRIT[cn](V), t, m, rng)
        good = (s == 1.0); ok &= good
        print(f"  [{'ok ' if good else 'FAIL'}] {cn:<10} finds a planted spike ({s:.2f})")

    W = ndimage.gaussian_filter(rng.normal(0,1,(40,40,40)).astype(np.float32), 2)
    mw = np.ones(W.shape, bool)
    for cn in ORDER:
        frac = n_distinct(CRIT[cn](W), mw) / W.size
        good = frac > 0.05; ok &= good
        print(f"  [{'ok ' if good else 'FAIL'}] {cn:<10} not degenerate ({100*frac:.0f}% distinct)")

    Z = np.zeros((20,20,20), np.float32)
    mz = np.ones(Z.shape, bool)
    tz = np.zeros(Z.shape, bool); tz.ravel()[:100] = True
    r1 = rank_score(Z, tz, mz, np.random.default_rng(1))
    exp = 100/Z.size
    good = abs(r1-exp) < 0.05; ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] all-tied input scores at chance "
          f"({r1:.3f} vs {exp:.3f})")

    Nz = rng.normal(0,1,(30,30,30)).astype(np.float32)
    mn = np.ones(Nz.shape, bool)
    tn = np.zeros(Nz.shape, bool)
    tn.ravel()[rng.choice(Nz.size, 500, replace=False)] = True
    s = rank_score(crit_hyper(Nz), tn, mn, rng)
    ch = 500/Nz.size
    good = s < ch*3; ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] hyper at chance on unrelated truth "
          f"({s:.4f} vs {ch:.4f})")

    print("  ALL PASSED\n" if ok else "  FAILED\n")
    return ok

if not self_test():
    sys.exit("Self-tests failed. Nothing was run. Fix the code first.")

# =====================================================================
files = sorted(f for f in os.listdir(DATA_DIR)
               if f.startswith("BRATS") and f.endswith(".nii.gz"))[:N_VOLUMES]
if not files:
    sys.exit(f"No volumes found in {DATA_DIR}")
print(f"Volumes to process: {len(files)}\n")

tie_rng = np.random.default_rng(999)
rows = []
DISTINCT = collections.defaultdict(list)
t_start = time.time()

for vi, fname in enumerate(files):
    rng = np.random.default_rng(SEED + vi)
    vol = nib.load(os.path.join(DATA_DIR, fname)).get_fdata()[..., MODALITY].astype(np.float32)
    brain = vol > 0
    if brain.sum() < 10000:
        print(f"  [{vi+1}] {fname}: too little brain, skipped"); continue
    R = float(vol[brain].max() - vol[brain].min())
    _, yy, xx = np.mgrid[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]
    t0 = time.time()

    for amp in AMPS:
        kx, ky = int(rng.integers(6, 20)), int(rng.integers(6, 20))
        ripple = (np.sin(2*np.pi*(kx*xx + ky*yy)/vol.shape[1]) * amp * R).astype(np.float32)
        I = vol.copy(); I[brain] += ripple[brain]
        true_err = np.abs(ripple)

        # truth excludes local extrema
        lo = ndimage.minimum_filter(I, 3); hi = ndimage.maximum_filter(I, 3)
        is_ext = ((I <= lo + 1e-6) | (I >= hi - 1e-6)) & brain
        truth = (true_err > 0.5*amp*R) & ~is_ext & brain

        for cn in ORDER:
            stat = CRIT[cn](I)
            sc = rank_score(stat, truth, brain, tie_rng)
            co, _ = spearmanr(stat[brain], true_err[brain])
            rows.append([fname, amp, cn, sc, co])
            if amp == AMPS[0]:
                DISTINCT[cn].append(n_distinct(stat, brain))

    el = time.time() - t_start
    print(f"  [{vi+1}/{len(files)}] {fname}  {time.time()-t0:.0f}s"
          f"   elapsed {el/60:.1f}m   eta {(el/(vi+1))*(len(files)-vi-1)/60:.1f}m")

with open("stripe_raw.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["volume", "amplitude", "criterion", "rank_score", "correlation"])
    w.writerows(rows)
print(f"\nWrote stripe_raw.csv  ({len(rows)} rows)")

# =====================================================================
S = collections.defaultdict(list); C = collections.defaultdict(list)
for v, a, cn, sc, co in rows:
    S[(float(a), cn)].append(sc); C[(float(a), cn)].append(co)

lines = []
def out(s=""):
    print(s); lines.append(s)

nv = len(set(r[0] for r in rows))
out("\n" + "="*80)
out(f"RESULTS   {nv} volumes, stripe artefact")
out("="*80)

out("\nDEGENERACY CHECK")
bad = False
for cn in ORDER:
    d = int(np.mean(DISTINCT[cn]))
    w = "  <-- DEGENERATE" if d < 1000 else ""
    if d < 1000: bad = True
    out(f"  {cn:<12}{d:>12,} distinct values{w}")
if bad:
    out("  WARNING: a degenerate criterion invalidates its own score.")

out("\nRANKING SCORE   mean (IQR)   truth excludes extrema")
out(f"{'amp':>6}" + "".join(f"{c:>22}" for c in ORDER))
for a in AMPS:
    cells = []
    for cn in ORDER:
        x = np.array(S[(a, cn)])
        cells.append(f"{x.mean():.3f} ({np.percentile(x,25):.2f}-{np.percentile(x,75):.2f})")
    out(f"{a:>6}" + "".join(f"{c:>22}" for c in cells))

out("\nCORRELATION WITH TRUE PER-VOXEL ERROR")
out(f"{'amp':>6}" + "".join(f"{c:>22}" for c in ORDER))
for a in AMPS:
    out(f"{a:>6}" + "".join(f"{np.mean(C[(a,cn)]):22.3f}" for cn in ORDER))

out("\n" + "="*80)
out("DOES THE TOPOLOGY CONTRIBUTE?   hyper vs each alternative")
out("="*80)
for a in AMPS:
    out(f"\n  amplitude {a}")
    h = np.array(S[(a, "hyper")])
    for cn in ["median", "C1_bound", "C2_mean", "C3_zscore"]:
        o = np.array(S[(a, cn)]); d = h - o
        if np.allclose(d, 0):
            v = "IDENTICAL"
        else:
            try:
                _, p = wilcoxon(h, o)
                rb = (np.sum(d>0) - np.sum(d<0)) / len(d)
                tag = "BETTER" if d.mean() > 0 else "worse "
                v = (f"{tag} diff {d.mean():+.4f}  p={p:.2e}  "
                     f"eff={rb:+.2f}  wins {int((d>0).sum())}/{len(d)}")
            except ValueError:
                v = "n/a"
        out(f"    vs {cn:<11} {h.mean():.3f} vs {o.mean():.3f}   {v}")

with open("stripe_summary.txt", "w") as f:
    f.write("\n".join(lines))
print("\nWrote stripe_summary.txt")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for cn, mk in zip(ORDER, ["o-","s-","^-","v-","d-"]):
        ax[0].errorbar(AMPS, [np.mean(S[(a,cn)]) for a in AMPS],
                       yerr=[np.std(S[(a,cn)]) for a in AMPS],
                       fmt=mk, capsize=3, label=cn)
        ax[1].errorbar(AMPS, [np.mean(C[(a,cn)]) for a in AMPS],
                       yerr=[np.std(C[(a,cn)]) for a in AMPS],
                       fmt=mk, capsize=3, label=cn)
    ax[0].set_xlabel("ripple amplitude"); ax[0].set_ylabel("ranking score")
    ax[1].set_xlabel("ripple amplitude"); ax[1].set_ylabel("correlation with true error")
    for a_ in ax: a_.grid(alpha=.3); a_.set_ylim(0, 1.05)
    ax[0].legend()
    plt.tight_layout(); plt.savefig("fig_stripe.png", dpi=200)
    print("Wrote fig_stripe.png")
except ImportError:
    print("matplotlib missing -- figure skipped")

print(f"\nTotal: {(time.time()-t_start)/60:.1f} minutes")
print("""
NEXT
  Read stripe_summary.txt.

  The claim stands only if hyper is BETTER than median AND better than
  all three controls, with consistent win counts. If any control
  matches hyper, that control is the honest baseline and the paper must
  say the topological framing adds nothing.
""")
