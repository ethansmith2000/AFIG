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

You can visualize the generation process by taking a finished image and running visualize_reconstruction found in utils.py, from which the frames can then be made a video with the moviepy function



## Future Plans
- Try some more on the GMM methods, see if we can fix the STDs at the std of the dataset or use a fixed per position learnable std
- Can we do this in latent space? I'd really like to somehow encode each concentric ring into a single token to make this scalable.
