Practical Guide of dimensionality reduction algorithms:
1. decision table:
2. Linear dimensionality reduction
	1. PCA
	2. FA
	3. CCA, ICA
	4. GFA
	5. GPFA
	6. mDLAG
2. non-linear dimensionality reduction (data visualization)
	1. t-SNE
	2. UMAP
---

# short descriptions

PCA 
1. finds a subspace that preserves the most variance of the data via SVD, then ordering by largest eigenvalue. 
2. It can be used also for whitening and removing collinearity by enforcing orthogonality in the axes. 
3. It assumes noise across latent dimension as isotropic. We arbitrarily determine cutoffs. 

FA
1. finds a low-rank approximation ($WW^T$) of the covariance of the data. 
2. The low-rank approximation is public noise, whereas the anisotropic noise covariance matrix is the private noise of each neuron, which generally follows a poisson spike. 
3. FA should be used when we want to understand shared or latent structure. 
4. We determine cutoffs via cross-validated log-likelihood

CCA

ICA

GPFA (X: (N_trials, N_neurons) -> X: (N_trials, n_components))
1. applies FA to time-binned spike count 
2. smoothes low-dimensional trajectories by fitting a gaussian process model

t-SNA

UMAP