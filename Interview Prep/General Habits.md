# Exploratory Data Analysis

Basics:
Usually, the data is shaped as \[n_trials, n_channels, n_samples\], where we are predicting `n_trials`. Here channels usually denotes an electrode.  
##### Initial Inspection
Suppose we get `X.shape=(n_trials, n_channels, n_samples)`, and $y$ vector.
```python
# X.shape=(n_trials, n_channels, n_samples)

# check shapes and data types
# should have 1 label per trial
X.shape, y.shape, X.dtype, y.dtype
assert X.shape[0] == len(y)

np.unique(y, return_counts=True) # classes and balance

np.isnan(X).mean(axis=(0, 2)) # check NaN fraction per channel

# per channel scale
X.std(axis=(0, 2)) # any dead or very noisey channels?

# trial average trace
plt.plot(X[:, ch].mean(axis=0))
```

Questions:
1. _data encoding:_ are there any specially encoded data? (less applicable for neural time-series)
2. What is the sampling rate? where is t=0? 
---
##### Missing data handling
3. _Missing completely at random:_ drop or impute
4. _Missing depends on other variables or value itself:_ Add missingness indicator or use model that handles missing data (sklearn's FA doesn't support NaN)
5. _Tabular defaults:_ If numerical, median imputation, and add missingness indicator. For categorical, if we determine features are correlated, IterativeImputer or KNNImputer. 
6. _Time Series:_ use linear interpolation for short gaps, if there are real-time or causal constraints, only forward fill. For longer gaps, we keep them marked missing and consider dropping the trials that overlap them. 
The imputer should be fitted in the training fold in the pipeline
---
##### OOD or high variance (time-series neuro)
if channel has 10x variance compared to others, first we inspect the trace and Power Spectral Density. We can also directly try to interpret from the [[Comp Neuro background#PSD|PSD]]. Note that the integral (0 -> 1/2 max sampling rate) of the PSD is the total variance of that channel. ****

Plotting the PSD
```python
from scipy import signal

# X.shape=(n_trials, n_channels, n_samples)

frequencies, psd_values = signal.welch(...)
```

a few common causes:
1. broadband noise
	1. _diagnosis:_ Uniformly elevated noise floor across the high-frequency spectrum ($>300\text{ Hz}$) on PSD; total signal variance is high without localized spectral peaks.
	2. _explanation:_ random, high-frequency electrical voltage fluctuations
	3. _solutions:_ drop the channel or interpolate from neighbors
2. line noise (electromagnetic interference)
	1. _diagnosis:_ Sharp, extremely narrow peak at $50\text{ Hz}$ or $60\text{ Hz}$ on PSD (plus harmonics at $120\text{ Hz}, 180\text{ Hz}$).
	2. _explanation:_ Electromagnetic interference coupled from nearby AC power grids or room electronics.
	3. _solution:_ Narrowband notch filter (e.g., $59.5\text{--}60.5\text{ Hz}$ FIR/IIR filter)
3. muscle artifacts 
	1. _diagnosis_ Broadband high-frequency bursts spanning $>20\text{--}200\text{ Hz}$
	2. _solution:_ - Low-pass filtering ($<30\text{ Hz}$ cutoff for ERPs), or canonical correlation analysis (CCA) / ICA, or reject Epoch
4. motion artifacts, baseline drift
	1.  _diagnosis_ std >> median absolute deviation
	2. _solution:_ reject affected epochs, high-pass filter (>0.5Hz), or apply Artifact subspace reconstruction (if EEG)
5. Bad / Disconnected Channels (Flatline or Saturation)
	1. *Diagnosis*: Near-zero time-domain variance ($\text{STD} \approx 0$) or clipped rail voltages; flat line across frequencies in PSD.
	2. *Solution:* Drop the channel (`raw.info['bads']`) and interpolate using spatial neighbors 

Workflow:
1. plot superimposed time traces.
	1. compare bad channel vs good neighbor channels, look for amplitude scale difference, low frequency drift
	2. compute welch's PSD by calculating overlapping windows to smooth out short term random noise
	3. compare PSD overlay, identify if the extra power is narrowband (single sharp peak) or broadband (raise baseline)
	4. apply targeted filtering
		1. if power is concentrated at specific frequency -> compare with ground connection. 
		2. if power is uniform across high frequencies -> hardware or measurement issues. 
##### Normalization and Standardization
We might z-score some features for a classifier and evaluate with CV. We have several sessions. 

$\mu, \sigma^{2}:$ computed training fold per channel. Use `StandardScaler`
We normalize per-session using the training data or the initial calibration data given in a real-trial. 
For a real-time optimization system, we can do a EMA or sliding window stats version.

##### How to subsample signals
You want to go from 1 kHz to 100 Hz. What must happen before you take every 10th sample?

take the new nyquist (100/2 Hz), low-pass the data through it by `signal.decimate(x, 10, axis=0)`
##### How to deal with Low-data
If lower rate, high risk of overfitting, therefore stronger regularization

else, reduce features by grouping time into windows (RMS), dimensionality reduction, or feature selection. 

More trials is always better
# Metrics
1. Accuracy: $\frac{n_\text{accurate}}{n_{\text{samples}}}$
2. precision: $\frac{TP}{TP+FP}$ Of all predicted positive, how many actually positive 
3. Recall: $\frac{TP}{TP+FN}$ Of all positive, how many got right?
4. Specificity: $\frac{TN}{TN+FP}$ Of all cases negative, how many got right?
5. F1: $\frac{\text{precision x recall}}{\text{precision + recall}}$ use for imbalanced data
6. ROC-AUC
7. PR-AUC: Area Under the Precision-Recall Curve, ignores true negatives, baseline is proportion positive class in dataset. Higher scores indicate better performance on minority class. 

Confusion matrix interpretation

