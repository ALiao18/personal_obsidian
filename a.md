# Tacit challenge brief: temperature session QC

**Task:** load 80 sessions, explore them, implement `MyCustomFilter(FilterStrategy)` in `filtering.py` (`fit` learns thresholds, `apply` returns a filtered **copy**), wire it into `main()` in `candidate_filter_script.py`, plot before vs after, and explain the criteria in comments. Optionally, add quality metrics. **No tests**, so you're judged on judgment, clean code and explanation. Only pandas, numpy and matplotlib are in `requirements.txt`. If you add scipy, add it there too.

## 1. Data flow and order

```
example_recordings.json → load_data() → df: 80 rows × [recording_id, room, serial_number, temperature(list[101]), time_points(list[101])]
  → compute_features(df) → F: 80 × [std, range, max_step, slope, mean]        (one row per session)
  → MyCustomFilter.fit(df): per-room median and MAD of features → thresholds (stored on self)
  → MyCustomFilter.apply(df): recompute features → keep rows where every robust z-score is within the limit → df.loc[mask].copy()
  → plot_final_data(df_original) vs plot_final_data(filtered_df) + print the removed sessions and why
```

|#|What|Where|By minute|
|---|---|---|---|
|0|`python candidate_filter_script.py` runs with DummyFilter|terminal|2|
|1|EDA cell: features table, plots by room and by sensor|notebook or script|8|
|2|`compute_features` helper + `MyCustomFilter.fit/apply`|`filtering.py`|20|
|3|swap DummyFilter for MyCustomFilter in `main()`, before vs after plots, removed list|script|25|
|4|explanatory comments, quality metrics, plot of flagged sessions|script|40|

Run commands: `uv venv && uv pip install -r requirements.txt`, then `uv run python candidate_filter_script.py`. Or use `python -m venv` as the README says.

## 2. Quick EDA (paste into a notebook cell or a Python REPL in the repo folder)

```python
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
df = pd.DataFrame(json.load(open("example_recordings.json")))
print(df.shape); print(df.dtypes); print(df.room.value_counts()); print(df.serial_number.value_counts())
print(pd.crosstab(df.room, df.serial_number))
print("lengths:", df.temperature.apply(len).unique(), df.time_points.apply(len).unique())
print("any NaN:", df.temperature.apply(lambda v: np.isnan(np.asarray(v, float)).any()).sum())

def feats(row):
    t = np.asarray(row["time_points"], float); x = np.asarray(row["temperature"], float)
    return pd.Series({"mean": x.mean(), "std": x.std(), "range": np.ptp(x),
                      "max_step": np.abs(np.diff(x)).max(), "slope": np.polyfit(t, x, 1)[0]})
F = pd.concat([df[["recording_id", "room", "serial_number"]], df.apply(feats, axis=1)], axis=1)
print(F.groupby("room")[["mean", "std", "max_step", "slope"]].median())
print(F.sort_values("max_step", ascending=False).head(10))

fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
for ax, (room, g) in zip(axes, df.groupby("room")):
    for _, r in g.iterrows():
        ax.plot(r["time_points"], r["temperature"], lw=0.8, alpha=0.6)
    ax.set(title=room, xlabel="time (h)"); axes[0].set_ylabel("temperature (a.u.)")
fig.tight_layout(); plt.show()
```

**What you'll see:**

- **Two flatlines near 0:** `20250127_10h39m` and `20250127_10h55m`, which look like a dead or disconnected sensor.
- **Sudden spikes** (max step 5–12 vs a normal ~1) in `20250124_*` in both rooms, and in `20250123_18h09m`.
- **SN 54** (`20241203_09h09m`) is the only session from that sensor and the only one that **warms**.
- **Room_0 varies more than Room_1**, which is why the thresholds are per room.
- **SN 59 reads about 2 units warmer.** That's a sensor _offset_, not an anomaly, so **don't filter on the mean** (if you do, it wrongly flags all the SN 59 sessions).

## 3. Pseudocode

### `compute_features(df) -> DataFrame` (module-level helper in `filtering.py`)

- for each row: `x = np.asarray(row.temperature, float)`, `t = np.asarray(row.time_points, float)`
- `std = x.std()`, `range = ptp(x)`, `max_step = max(|diff(x)|)` (spikes), `slope = polyfit(t, x, 1)[0]` (trend), `mean`
- return a DataFrame with the **same index as df**, plus `room`
- gotcha: the columns hold lists, so convert them with `np.asarray(..., float)`

### `MyCustomFilter.__init__(self, z_thresh=5.0, features=("std", "max_step", "slope"), group_col="room")`

- store the parameters; set `self.median_ = self.mad_ = None`

### `fit(self, data)`

- `F = compute_features(data)`
- `self.median_ = F.groupby(group_col)[features].median()`
- `self.mad_ = 1.4826 * F.groupby(group_col)[features].agg(lambda s: (s - s.median()).abs().median())`, scaled to match a Gaussian std; guard against 0 by replacing it with a small number
- why robust stats: the anomalies themselves would inflate a mean/std threshold
- `return self`

### `_flag(self, data)` (helper returning a reasons Series)

- `F = compute_features(data)`
- `z = (F[features] - self.median_.loc[F[group_col]].values) / self.mad_.loc[F[group_col]].values`
- `reasons` = the names of the features where `|z| > z_thresh`, joined into a string per row
- add an absolute **flatline rule**: `std < 0.1` or `|mean| < 1` gives a "flatline" reason, because a dead sensor is the clearest fault

### `apply(self, data)`

- `reasons = self._flag(data)`; `keep = reasons == ""`
- print the number kept and removed, and the removed `recording_id`s with their reasons
- `return data.loc[keep].copy()`, i.e. a **new** DataFrame, as the docstring requires
- if `fit` wasn't called (`self.median_ is None`), raise a clear error

**Expected result** (I tested this logic with z > 5): 11 sessions removed = the 4 Room_1 spike sessions from Jan 24 + the 2 flatlines + SN 54 + `20250122_09h56m` + `20250124_17h47m`, `20250124_18h08m`, `20250123_18h09m`. 69 kept.

### `main()` changes

- `from filtering import MyCustomFilter`; `my_filter = MyCustomFilter(z_thresh=5)`
- `plot_final_data(df_original, "Original")`, then `plot_final_data(filtered_df, "MyCustomFilter")`
- optionally, plot only the removed sessions with their reasons in the legend
- add comments explaining each criterion and why (spike, flatline, trend; per room; no mean)

### Optional quality metrics (part 4)

- per session: `std`, `max_step`, the **noise estimate** `std(diff(x)) / sqrt(2)` (high-frequency jitter), residual std around a linear or smoothed trend, and the number of steps above the threshold
- compare kept vs removed: a table or box plot of each metric, split by the filter's verdict

## 4. Things to bring up

- **What counts as an anomaly:** sensor faults (flatline, spikes, a sensor nobody else uses) vs real-but-unusual temperature behavior. Ask which one they care about.
- **Robust, per-group thresholds:** median/MAD per room, so normal differences between rooms aren't flagged. With only 80 sessions, avoid heavy ML.
- **The SN 59 offset is calibration, not an anomaly.** I deliberately didn't filter on the mean.
- **SN 54 has n = 1.** I can't tell a faulty sensor from an odd session. I flagged it, but would ask.
- **Fit vs apply:** thresholds are learned once and reused on new sessions. If the training data contains many anomalies, the robust stats still hold.
- **The threshold trade-off:** z = 5 is conservative. Show how many sessions get removed at z = 3, 4, 5 if you have time.
- **Alternatives** given more time: Isolation Forest or PCA reconstruction error on the feature vectors, changepoint detection for jumps, Hampel-filter _repair_ of spikes instead of dropping the whole session.

## 5. Cheat sheet for this repo

```python
x = np.asarray(row["temperature"], float); t = np.asarray(row["time_points"], float)   # lists → arrays
df.apply(fn, axis=1)                     # fn(row) → pd.Series → one column per key
np.ptp(x); np.abs(np.diff(x)).max(); np.polyfit(t, x, 1)[0]                           # range, max jump, slope
g = F.groupby("room")["std"]; g.median(); g.agg(lambda s: (s - s.median()).abs().median())   # robust per group
med.loc[F["room"]].values                # broadcast each group's stats to its rows
df.loc[mask].copy(); df.index.difference(kept.index)                                  # filtered copy; removed ids
for room, g in df.groupby("room"): ...   # loop over groups
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
```

**One-liner for the Q&A:** _"I summarized each session with fault-sensitive features (spikiness, variability, trend). I learned robust per-room baselines in `fit`, and in `apply` I remove sessions that are extreme on any feature, or flat. I didn't use the mean, because the sensor offset is calibration, not a fault."_