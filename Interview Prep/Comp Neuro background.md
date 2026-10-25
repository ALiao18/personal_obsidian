# PSD

PSD plot transforms a time-series voltage signal into frequency domain: allows interpretation on how signal energy is distributed across frequencies. 

x-axis: frequency in Hz 
y-axis: $\frac{\mu V^{2}}{Hz}$ energy per unit frequency, oftentimes we use $\frac{dB}{Hz}$ since log-scale is preferred for data that spans multiple OOMs.

We can read total variance from the PSD:
$$
\begin{align}
\sigma^{2}=\int_{0}^{f_{s}/2} PSD(f) \quad df
\end{align}
$$
# General techniques
##### spike binning
