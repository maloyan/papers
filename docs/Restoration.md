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
- [A Survey on Leveraging Pre-trained Generative Adversarial Networks for Image Editing and Restoration](https://arxiv.org/pdf/2207.10309.pdf)
- [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://www.youtube.com/watch?v=Lg97gWXsiQ4)

## Image-Super-Resolution

- [SRCNN](configs/restorers/srcnn/README.md) (TPAMI'2015)
- [SRResNet&SRGAN](configs/restorers/srresnet_srgan/README.md) (CVPR'2016)
- [EDSR](configs/restorers/edsr/README.md) (CVPR'2017)
- [ESRGAN](configs/restorers/esrgan/README.md) (ECCV'2018)
- [RDN](configs/restorers/rdn/README.md) (CVPR'2018)
- [DIC](configs/restorers/dic/README.md) (CVPR'2020)
- [TTSR](configs/restorers/ttsr/README.md) (CVPR'2020)
- [GLEAN](configs/restorers/glean/README.md) (CVPR'2021)
- [LIIF](configs/restorers/liif/README.md) (CVPR'2021)

## Generation

- [CycleGAN](configs/synthesizers/cyclegan/README.md) (ICCV'2017)
- [pix2pix](configs/synthesizers/pix2pix/README.md) (CVPR'2017)

Methods
--------------------------------------------

1. Gan inversion - find vector in latent space corresponding to the original image. Then move that vector towards more quality image

![](https://images.deepai.org/converted-papers/1907.10786/x6.png)

2. 

Datasets
--------------------------------------------
- [vqgan 16k reconstruction](https://huggingface.co/datasets/johnowhitaker/vqgan16k_reconstruction)
