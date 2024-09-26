# GeoLDM
In the domain of conditional image generation, diffusion models are the state-of-the-art. The applications of diffusion models are many, extending beyond images to speech and even video generation. Although there have been previous advancements with geo-transformations and implementations of Bicycle GANs for related tasks, there is a notable absence of Latent Diffusion Model (LDM) applications tailored for generating street view imagery from satellite data. Here, we present GeoLDM, an LDM that models the conditional distribution of Google Street View images given a set of corresponding geospatial features at that location. Further, we incorporate a pre-trained model released by SatlasAI for feature extraction and further conditioning of our model. To adapt the LDM to our geospatial task, we implement a variational autoencoder for ground image data and an interpolation head for satellite image embeddings to feed into a Denoising Diffusion Probabilistic Model (DDPM). Our method is capable of generating highly plausible street view images, and incorporates new features into the typical LDM framework as presented in previous works.

<img width="800" alt="geo3" src="https://github.com/Praccho/GeoLDM/assets/39883887/fdce4d2d-d7ed-400a-b593-27378915f2fa">
