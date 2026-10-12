##### 8.26.2026
_Update:_ Received official confirmation from Byron Yu for advisement. 
_Meeting Notes_
1. Finish HW8, which covers PCA, pPCA, and FA
2. Most implementations uses eigenvalue decomposition (sklearn). This implementation will use EM algorithms. I am to implement them, and share them in a private repo with Evren
3. asked Byron about the BCI-sim project with Chase. Response:
	1. the latents project would be great training for the BCI-sim project. All those algorithms are the bread and butter
	2. Conditional on doing a great job with latents, transition to BCI-sim project after 1-2 semesters.
4. _Advice:_ Sometimes we let the things that have deadlines supercede the priority of things that don't have a deadline even though those without a deadline are often more important.

Goal by next Friday
1. Finish HW8 by Wed, send to Evren for Review
2. fully understand the repo structure and prepare all dependencies and environments to implement

__8.30.2026__
Some implementation notes before getting started:
1. never get the full $D \times D$ matrix out, instead, use something called the woodbury identity
2. since all matrices to be inverted are PSD by definition, use cholesky solvers. 
3. to avoid variance from numerical instability around 0, clip to small positive floor after every M step and add diagonal jitter before cholesky call as defensive default
4. EM guarantees monotonical non-decreasing loglikelihood in each iteration, so test log likelihood. Track both log likelihood as well as parameter change. 
5. Since PCA is unstable to sign flips (figure out why), when checking numerical values, align via procrustes/SVD before comparing
6. Standard practice for EM initialization: initialize from the top-k PCA eigenvectors/values and diagonal noise from residual variacne per feature after projecting out those k components
7. always mean center per feature before fitting. There is some more nuance to be discussed for FA specifically, regarding the choice of z-score normalization or not. 

Upon doing hw8, Yu's class implementation uses scatter matrix $S=X^TX$ whereas sklearn uses the sample covariance matrix. 
	In signal processing, 

##### 9.4.2026
1. request my own branch for latents (I see that there is an independent branch called mdlag, preference for 1 branch per algo?)
2. `GFAModel` exposes `.obs_posterior` and `.latents_posterior`. FA is non-Bayesian, so there is no observation posterior — `architecture.md` sketches `obs_params: ObsParamsPoint`. Do you want `FAModel.obs_params` (semantically honest, breaks uniformity) or something else? This affects whether generic downstream code can treat models interchangeably.
3. choosing dimensionality: should we also include cross validation since for FA there is no auto relevance determination. 
4. pPCA and FA are the same package with a flag difference (noise: isotropic boolean)

###### goal for the semester
cross validated dimensionality selection (FA, pPCA< pCCA, GPFA)
`latents.fa` merged to `main`, covering FA + pPCA as one module:**
- `src/latents/fa/`: `config.py` (`FAFitConfig`, `FASimConfig`), `model.py` (`FAModel`), `inference.py` (EM: `fit`, `init_posteriors`, `infer_latents`, `infer_loadings`, `infer_obs_mean`, `infer_obs_prec`, `compute_loglik`), `simulation.py`, `tracking.py`, `analysis.py`
- pPCA as `isotropic: bool` on `FAFitConfig`, not a separate subpackage
- `tests/test_fa/` mirroring the source tree, including parameter-recovery and log-likelihood-monotonicity tests
- Docs: `reference/models/fa/*.rst`, `user_guide/methods/fa.md`, gallery scripts in `examples/fa/`
- `CHANGELOG.md` entry, released as part of a version bump (v0.0.6 or similar)

**`latents.pcca`** — two-group extension of FA. `ObsStatic` already handles multi-group via `dims`; `ObsParamsPoint` and `LatentsPriorStatic` already fit. Should be materially cheaper than FA once FA's pattern is established, because it's the same skeleton with a different `y_dims` shape.


class FAModel:
    latents_prior: LatentsPriorStatic        # reuse
    obs_params: ObsParamsPoint               # point estimates, NOT a posterior
    latents_posterior: LatentsPosteriorStatic # reuse — a real posterior

pPCA
	static prior, 
	point-estimate parameters, 
	one group, 
	isotropic noise

##### 9.11.2026
Updates:
1. completed mini-latents (implementation of PCA, pPCA, FA) in numpy
2. validated via Claude generated tests and comparisons against sklearn

Meeting notes:
1. writing the full stack (tests, UI/API, documentation, integration with existing tools/utilities) is equally as time consuming as understanding the core algorithms. 
2. expectations:
	1. For this semester, it would already be **excellent** to have production grade PPCA, FA fully integrated into latents. 
	2. defer time-series methods
	3. start depth first with pPCA as full stack example to learn
	4. "the wringer"
		1. be able to simulate/generate data from these models
		2. use these data for testing and validation, to understand when methods recover ground truth and when they fail
		3. using current implementations, generate synthetic data with known ground truth. Systematically study
			1. recovery as a function of n trials, n neurons, etc.
			2. failure modes/adversarial cases when these methods fail
				1. look at Evren's supplemental figures in the DLAG paper. 
				2. look at existing latents benchmarks on the web for convergence behavior, performance curves, etc. 
3. logistics
	1. I will be invited as a collaborator on the repo
	2. for exploratory or early work, it may be better to work in a separate repo, and only integrate into latents when its ready
	3. current implementation is in numpy, future should try to move towards a pyTorch implementation
4. AI Coding
	1. _Encouraged:_ unit tests, coverage, boilerplate code, input validation, documentation (docstrings, consistency), code review for style and testing gaps
	2. _discouraged:_ core algorithms
5. learn to make good presentations for updates. Should start with a recap

Action items b4 next meeting
1. share mini-latents with evren
2. start porting pPCA into latents structure, depth first
3. take "pPCA, FA" into the wringer. Exploratory. Read through Evren's DLAG supp

##### 9.24.2026
Updates
- W initialization
	- **Random:** cheap, O(dk), but starts far from the solution, so it can mean more iterations or a different local optimum. Never start at W = 0: EM stays stuck there.
	- **ML init:** O(Nd²) to build S, plus O(d³) for `eigh`. It runs **once**, so it pays off if it saves iterations. For large d, a truncated SVD costs O(Ndk).
	- **Scaling W = U_k(Λ_k − σ²I)^½:** the variance along each top direction is signal plus noise, so the latent should get only λᵢ − σ². Scaling by √λᵢ counts the noise twice.
	- For pPCA this is the exact ML solution. For FA it's a good starting guess.
	- Open: pPCA's ψ is still initialized to the total variance, not σ².
- Noise as vectors
	- `noise_as_vec()` returns the (d,) diagonal; `as_matrix()` is derived from it.
	- E-step: O(d³) → O(dk). No d×d noise matrix anywhere during EM.
	- Shape guard: raises an error if a dense Ψ is passed.
	- FA noise update, using sufficient statistics: O(Ndk²) → O(dk²).
##### 10.2.2026
updates:
1. E-step now outputs sum of second moments (k, k) instead of just second moments (N, k, k)
2. split `_sym_inv_lodget` into cholesky and triangular solves from the diagonal. 
3. use woodbury identity and determinant lemma to calculate log llikelihood
4. calculate $X^2$ once, and pass as variable. 
5. $X^T Z$ is used in cross term. We calculate it once in fit and pass it as a param
questions:
6. what should we name inverse transform? It computes E[y | x] = C x + d.
	1. could use something like conditional_mean
notes
7. For tests, ultimately we want a full suite for API, functions, classes, rejector accept, expected data types. covering it all
8. algorithms should be optimized such that they scale linearly wrt trials and neurons, while letting the latent variables be the bottleneck. 
9. range of likely use cases: 10s-1000s neurons, 100s-1000s sessions/trials
10. start thinking about simulations!! Think about those models are generative models instead of fitting models. 
11. start thinking about cross validation: selection of the dimensionality, and suppose we have the correct dimensionality, how well do we recover from that?
12. we have the option of choosing between EM or ML solution for PCA. We want a "smart" algorithm to pick which we can fit based on memory/time constraints given N, d, k. benchmark on Yu group CPU cluster. 
##### 10.9.2026
1. finished writing the script for comparing PCA: closed-form ML solution vs EM
	1. _ML_: covariance $O(N d^2)$ + full eigh $O(d^3)$, via pPCA.fit(method="closed_form").
	2. _EM_: $W \sim \mathcal{N}(0, 1)$, $\sigma^2$ = mean feature variance, then EM with the default stopping rule, $O(Ndk)$
	3. _NEXT_WEEK_ benchmark on CPUs. 
2. enforce sign stability in PCA, pPCA and FA
	1. eigenvectors and eigenvalues are computed via `np.linalg.eigh`. sign is unstable and dependent on hardware
	2. _fix:_ Flip each column of W (d,k) so its largest |entry| is positive.
	3. _Test:_ in every column of `components_`, the largest-magnitude entry is positive.
3. E-step output
	1. _prev_: output Ezz, (N, k, k), but m_step uses sum over N
	2. _now_: output sum_Ezz (k, k)
	3. `sum_Ezz` =`N * M + Ez.T @ Ez` 
		1. # sum over samples of each sample’s second moment E[z zᵀ]
4. tests plan 
	1. Convergence: done
		1. _(DONE)_`test_fit`: `converged`, not `decreasing_ll`, iteration count matches a regression baseline (pPCA and FA)
		2. _(DONE)_ `test_ll_monotonicity`: `ll_history_` from the actual `fit()` is non-decreasing within tol
	2. Woodbury identity vs dense inverse (log likelihood)
		1. _(DONE)_ `test_e_step_matches_woodbury_form` : E-step posterior mean equals `X @ inv(C) @ W`, with C = WWᵀ + Ψ built densely
		2. _(DONE)_ `test_log_likelihood_matches_scipy`: (Woodbury + determinant lemma) equals `multivariate_normal(cov=C).logpdf`
		3. test quadratic term Σₙ xₙᵀC⁻¹xₙ = `np.linalg.inv(C)`
			1. would need to separate out quadratic term as helper
	3. ML pPCA = stationary point in EM pPCA
		1. _Test_: start from the closed form and run one EM step. Check implied covariance WWᵀ + σ²I and σ² don’t change (`rtol` ≈ 1e-10, floating point only).
		2. _Oracle_: the closed form is computed independently with `np.linalg.eigh`
	4. Parameter recovery (`@pytest.mark.fit`)
		1. Simulate at N = 500, 5,000 and 50,000; the error in the implied covariance and in ψ should shrink roughly 1/√N
		2. do not test W because W is free to rotate
	5. Invariances
		1. FA is equivariant to per-feature rescaling; pPCA is not; FA reduces to pPCA when the noise is isotropic
	6. Posterior covariance
		1. `(sum_Ezz − EzᵀEz) / N` equals posterior covariance of z $M = (I + WᵀΨ⁻¹W)⁻¹$, computed with `np.linalg.inv`
		2. check M is symmetric and PD
	7. M-step is a stationary point of Q (∂Q/∂W ≈ 0, ∂Q/∂ψ ≈ 0)
		1. M-step should return its exact maximizer
		2. Procedure:
			1. run 1 EM step
			2. compute Q with an independent dense helper (_compute_q, textbook posterior, does not call em_core)
			3. nudge each parameter slightly up and down and measure slope of Q
			4. compare it to the slope at the starting point

NEXT_WEEK_PLAN
1. algorithmic:
	1. variance floor: M-step it clips ψ to `varFloor = minVarFrac · diag(cov(X))` (default `minVarFrac` = 0.01) (haywood cases)
	2. EM only needs the data through the sample covariance S = XcᵀXc/N (O(d²k))
2. implement tests above
3. simulations: simulations.py, 
	1. test speed of EM and ML on cluster
	2. draft "smart" pPCA optimizer
4. extra: implement randomized linalg solver or find method to not do full eigendecomposition

