# Live-coding cheat sheet (patterns to retype, not import)

## Clock and habits
- **Check the clock at 15 / 25 / 35 min.** By 25 something should run end to end: load → features → model → number.
- Write placeholders as `raise NotImplementedError`, never half-finished lines, so the module still imports. **Run the tests after every function.**
- **Pass `axis=` every time.** Write the shape next to every array: `# (time, ch)` or `# (trials, ch, time)`.
- If you've been stuck for 5 min: write a crude version, say "quick version now, I'll come back", and move on.
- Not sure? **Reason from how it works, or run a 5-second test.** Never guess.

## 1. Inspect (first 2 minutes)
```python
x.shape, x.dtype, fs                          # what is each axis?
np.unique(labels, return_counts=True)         # per group! (case/typo mismatches, imbalance)
np.isnan(x).mean(axis=0)                      # NaN fraction per channel
np.nanstd(x, axis=0)                          # per channel, per group: dead (~0) or noisy (huge)
f, P = welch(x[~np.isnan(x).any(axis=1)], fs=fs, nperseg=2048, axis=0); plt.semilogy(f, P)   # line noise? band?
```
Skim the provided helpers too (`viz.py`, `io.py`).

## 2. Units
```python
n = int(round(ms * fs / 1000))                # ms → samples (ints; round, don't truncate)
t = np.arange(n_samples) / fs                 # sample index → seconds
```

## 3. Contiguous runs (NaN gaps, thresholds, events)
```python
pad = np.concatenate(([False], mask, [False])).astype(np.int8)
d = np.diff(pad); starts = np.flatnonzero(d == 1); ends = np.flatnonzero(d == -1)   # [start, end)
```
**Linear interpolation, per channel:**
```python
idx = np.arange(len(x)); bad = np.isnan(x[:, ch])
x[bad, ch] = np.interp(idx[bad], idx[~bad], x[~bad, ch])
```

## 4. Filtering (x is (time, ch))
```python
from scipy import signal
b, a = signal.iirnotch(w0=60, Q=30, fs=fs);  x = signal.filtfilt(b, a, x, axis=0)
sos = signal.butter(4, [20, 450], btype="bandpass", fs=fs, output="sos");  x = signal.sosfiltfilt(sos, x, axis=0)
# NaN-safe: x[long] = 0 → filter → x[long] = np.nan   (one NaN poisons an IIR filter's whole output)
# real-time: signal.sosfilt (causal, has lag); filtfilt uses future samples
# downsample: signal.decimate(x, q, axis=0)  (anti-aliasing filter built in), never x[::q]
```

## 5. Epoch and windowed features
```python
i0 = int(round(tmin * fs)); n_t = int(round((tmax - tmin) * fs))
ep = np.stack([x[o + i0 : o + i0 + n_t].T for o in onsets])          # (trials, ch, time)
ok = ~np.isnan(ep).any(axis=(1, 2))                                   # drop trials touching gaps; REPORT how many

win, step = int(round(win_ms * fs / 1000)), int(round(step_ms * fs / 1000))
nw = 1 + (ep.shape[-1] - win) // step
rms = np.stack([np.sqrt(np.mean(ep[..., k*step : k*step+win] ** 2, axis=-1)) for k in range(nw)], axis=-1)  # (trials, ch, nw)
X = np.log(rms + 1e-8).reshape(len(rms), -1)                         # (trials, ch*nw)
```
To choose the window: plot the trial-averaged envelope (use the **median** over trials, which is robust to artifacts). The response should sit inside the window.

## 6. Pool groups
```python
Xs, ys, gs = [], [], []
for s in sessions:
    ...; Xs.append(X_s); ys.append(np.char.lower(lab_s)); gs.append(np.full(len(X_s), s.name))
X, y, groups = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)
```

## 7. Evaluate
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, GroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))     # fit-on-train lives INSIDE the pipeline
# p ≈ n or p > n: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

cv_in = StratifiedKFold(5, shuffle=True, random_state=0)
cross_val_score(model, X[m], y[m], cv=cv_in, scoring="balanced_accuracy")            # (a) within one group (m = groups == g)
cross_val_score(model, X, y, groups=groups, cv=LeaveOneGroupOut(), scoring="balanced_accuracy")   # (b) unseen group
cross_val_score(DummyClassifier(strategy="most_frequent"), X, y, groups=groups, cv=LeaveOneGroupOut(), scoring="balanced_accuracy")  # baseline
cross_val_score(model, X, np.random.permutation(y), groups=groups, cv=LeaveOneGroupOut())  # leak check → ~chance
y_hat = cross_val_predict(model, X, y, groups=groups, cv=LeaveOneGroupOut())
ConfusionMatrixDisplay.from_predictions(y, y_hat, normalize="true")
```
**Per-group normalization** (cross-session fix; uses unlabeled data from that group, so say the assumption out loud):
```python
for g in np.unique(groups):
    m = groups == g;  X[m] = (X[m] - X[m].mean(axis=0)) / (X[m].std(axis=0) + 1e-8)
```
**Regression:** `Ridge(alpha=…)`, `scoring="r2"`. For time series, split by trial or block (`GroupKFold`), never by random bins.

## 8. Dimensionality reduction
```python
from sklearn.decomposition import PCA, FactorAnalysis
Xs = StandardScaler().fit_transform(X)
cum = np.cumsum(PCA().fit(Xs).explained_variance_ratio_);  np.searchsorted(cum, 0.9) + 1      # dims for 90% variance
ll = [cross_val_score(FactorAnalysis(k), Xs, groups=groups, cv=GroupKFold(4)).mean() for k in range(1, 21)]  # CV log-lik
best_k = 1 + int(np.argmax(ll))                                                                 # PCA(k).score = PPCA log-lik
fa = FactorAnalysis(best_k).fit(Xs); sh = (fa.components_**2).sum(0); pct_shared = sh / (sh + fa.noise_variance_)
Z = PCA(2).fit_transform(Xs)          # plot Z[y == c] per class
make_pipeline(StandardScaler(), PCA(10), LogisticRegression(max_iter=2000))   # DR for decoding goes INSIDE the pipeline
```

## Talking points (one line each)
- **Leakage:** anything fit on data (scaler, imputer, PCA) is fit on the training fold only, which a Pipeline inside cross-validation does for you.
- **Split by the unit you'll deploy across** (session, subject). Within-session scores are optimistic.
- **PCA vs FA:** PCA assumes equal noise per feature, so a noisy feature can dominate. FA separates shared from private variance, so prefer it for population and latent structure. Choose dimensionality by CV log-likelihood.
- **filtfilt** is zero-phase but non-causal. On a device, use `sosfilt` and accept some lag.
- **NaN** propagates through IIR filters. Fill temporarily, filter, then re-mask.
- A **too-good score** usually means leakage. Run the shuffled-label check.
- **Imbalanced classes:** use balanced accuracy and a confusion matrix, and compare against the Dummy baseline.

## Asking for help
- **AI:** "Implement exactly per this docstring; simple loop; ints via round(ms*fs/1000)." Paste the function. Read every line, then run the test.
- **Interviewer:** "I want shape X → Y. My plan is …. Does that sound reasonable, or is there something in the repo I should use?"
