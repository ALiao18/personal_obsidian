Why do you want to come to TreeHacks 2027?*

_Tell us about a project that you're proud of, technical or non-technical._

Neuroscientists record hundreds of neurons at once, and the field increasingly believes this activity lives on a few latent dimensions. Finding them reliably takes latent variable models, but existing tools are often slow or fragile.

I’m building latents, a dimensionality reduction library for CMU’s Neural Signal Processing group, aiming to beat sklearn on speed and memory for sparse neural data behind a simple API. I implemented PCA, probabilistic PCA, and factor analysis from scratch on a shared EM core, accelerated with the Woodbury identity and Cholesky factorizations.

The numerical optimization was a fun rabbit hole, but API design caught me off guard: a model’s equations don’t map cleanly onto code, and abstracting away layers of the optimization algorithm into modules, and providing an easy interface while allowing for deeper 

Optimization of pPCA and FA is nearly done; next are time-series methods like GPFA and DLAG. 

_What's something you want to work on in the next 10 years? Dream big!_

_Tell us a fun fact about you :)_

_Tell us about an experience you enjoyed, mundane or crazy. Give us a slice of your life!_

_Links that you'd like us to check out! GitHub, portfolio, LinkedIn, Devpost, deployed projects, awards, etc. Show us your coolest work!_

Felix Su, ys4858@nyu.edu

