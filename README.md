# AFIG (Autoregressive Fourier Image Generation)

blog post explainer: https://www.ethansmith2000.com/post/mimicking-diffusion-models-by-sequencing-frequency-coefficients

This repo contains 3 types of training for this method:
- train_mixture_cov.py - GMM with covariance predictions
- train_mixture_unroll.py - univariate GMM
- train_quantized.py - quantized the range of values into a discrete vocabulary

getting set up and training should be as easy as
```bash
pip install -r requirements.txt
python train_[method].py
```