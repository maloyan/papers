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
Some text

Overview
--------------------------------------------
- [A Survey on Leveraging Pre-trained Generative Adversarial Networks for Image Editing and Restoration](https://arxiv.org/pdf/2207.10309.pdf)
- [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://www.youtube.com/watch?v=Lg97gWXsiQ4)
- [High-Quality Pluralistic Image Completion via Code Shared VQGAN](https://arxiv.org/pdf/2204.01931.pdf) (They used Tfill)

### Image-Super-Resolution

- [SRCNN](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srcnn/README.md) (TPAMI'2015)
- [SRResNet&SRGAN](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srresnet_srgan/README.md) (CVPR'2016)
- [EDSR](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/edsr/README.md) (CVPR'2017)
- [ESRGAN](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/esrgan/README.md) (ECCV'2018)
- [RDN](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/rdn/README.md) (CVPR'2018)
- [DIC](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/dic/README.md) (CVPR'2020)
- [TTSR](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/ttsr/README.md) (CVPR'2020)
- [GLEAN](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/glean/README.md) (CVPR'2021)
- [LIIF](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/liif/README.md) (CVPR'2021)
- [Diverse Similarity Encoder for Deep GAN Inversion](https://arxiv.org/pdf/2108.10201.pdf)

### Generation

- [CycleGAN](https://github.com/open-mmlab/mmediting/tree/master/configs/synthesizers/cyclegan/README.md) (ICCV'2017)
- [pix2pix](https://github.com/open-mmlab/mmediting/tree/master/configs/synthesizers/pix2pix/README.md) (CVPR'2017)

### CVPR 2022
- [GCFSR: a Generative and Controllable Face Super Resolution Method Without Facial and GAN Priors](https://openaccess.thecvf.com/content/CVPR2022/papers/He_GCFSR_A_Generative_and_Controllable_Face_Super_Resolution_Method_Without_CVPR_2022_paper.pdf)
- [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf)
- [Deep Generalized Unfolding Networks for Image Restoration](https://openaccess.thecvf.com/content/CVPR2022/papers/Mou_Deep_Generalized_Unfolding_Networks_for_Image_Restoration_CVPR_2022_paper.pdf)
- [All-In-One Image Restoration for Unknown Corruption](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.pdf)

### Tools
- [MMediting tool](https://github.com/open-mmlab/mmediting)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)
- [StyleGAN XL](https://github.com/autonomousvision/stylegan_xl)

Methods
--------------------------------------------

1. Gan inversion - find vector in latent space corresponding to the original image. Then move that vector towards more quality image

![](https://images.deepai.org/converted-papers/1907.10786/x6.png)

2. 

Datasets
--------------------------------------------
- [vqgan 16k reconstruction](https://huggingface.co/datasets/johnowhitaker/vqgan16k_reconstruction)

Metrics & Loss
--------------------------------------------
- PSNR
- SSIM
- FID
