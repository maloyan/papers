Image Restoration (VQGAN)
=================================

Image with artifacts            |  Reconstructed image
:-------------------------:|:-------------------------:
![](https://datasets-server.huggingface.co/assets/johnowhitaker/vqgan16k_reconstruction/--/johnowhitaker--vqgan16k_reconstruction/train/27/reconstruction_256/image.jpg)  |  ![](https://datasets-server.huggingface.co/assets/johnowhitaker/vqgan16k_reconstruction/--/johnowhitaker--vqgan16k_reconstruction/train/27/image_256/image.jpg)

Abstract
----------------------------------------------

Photorealistic image generation is taking huge place in computer vision. We saw Dalle-2, Imagen and many others. The problem is they are huge and hard to reproduce, so people started to train smaller models for their needs like dalle-mini. The problem with those models is that they are not photorealistic enoght and you can see many artifacts on images. In this paper we are presenting model that is working on that kind of output with artifacts and improving its quality

Introduction
--------------------------------------------

Overview
--------------------------------------------
- [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://www.youtube.com/watch?v=Lg97gWXsiQ4)


Methods
--------------------------------------------

1. Gan inversion - find vector in latent space corresponding to the original image. Then move that vector towards more quality image

![](https://images.deepai.org/converted-papers/1907.10786/x6.png)

2. 

Datasets
--------------------------------------------
- [vqgan 16k reconstruction](https://huggingface.co/datasets/johnowhitaker/vqgan16k_reconstruction)