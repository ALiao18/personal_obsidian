usually, sampling rate is set as `fs`
Lets say we have `# X.shape = (n_trials, n_channels, n_samples)`

`detrend`
- _function:_ remove a constant offset or linear drift
- _Usage:_ 
	- `x_centered = signal.detrend(X, type='constant`