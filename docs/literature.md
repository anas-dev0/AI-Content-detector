# Literature: verified papers per track

Checked on 2026-11-10 against arXiv / CVF / proceedings pages. "Venue" is given only where confirmed. **UNVERIFIED** = the bibliographic details could not be confirmed from a primary page during this search, even if the paper is well known. We do not quote any paper's numbers in our results; they are context only.

## Track 1: frozen foundation-model features + linear probe

| Paper | Link | What we take from it |
|---|---|---|
| Ojha, Li, Lee. *Towards Universal Fake Image Detectors that Generalize Across Generative Models* (UnivFD), CVPR 2023 | https://arxiv.org/abs/2302.10174 · code https://github.com/Yuheng-Li/UniversalFakeDetect | The core recipe: a CLIP ViT-L/14 feature space **not trained for real/fake**, with a linear probe or nearest neighbour. Also the argument that ImageNet-only encoders do not generalise. |
| Radford et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), 2021 | https://arxiv.org/abs/2103.00020 | The backbone and its preprocessing/normalisation. |
| Cozzolino et al. *Raising the Bar of AI-generated Image Detection with CLIP*, 2023 (CVPRW 2024) | https://arxiv.org/abs/2312.00195 | Evidence that CLIP features work with very few training examples. Motivates our equal-data comparison at 10k. |
| Zhou et al. *Simplicity Prevails: The Emergence of Generalizable AIGI Detection in Visual Foundation Models*, arXiv 2026 (preprint) | https://arxiv.org/abs/2602.01738 | Reports that linear probes on frozen DINOv3 / Perception Encoder / MetaCLIP 2 features are a very strong baseline, with stated weaknesses under recapture/transmission. Justifies adding a **DINOv3 probe** and testing the screenshot chain. |
| Siméoni et al. *DINOv3*, 2025 | https://arxiv.org/abs/2508.10104 · weights https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m (gated, DINOv3 License) | Self-supervised backbone (ViT-S/B/L/H+/7B, patch 16) for the second probe and a LoRA candidate. Licence terms for commercial use: **UNVERIFIED**. Fine for research. |

## Track 2: LoRA fine-tuning of a pretrained backbone

| Paper | Link | What we take from it |
|---|---|---|
| Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*, 2021 | https://arxiv.org/abs/2106.09685 | The adapter method (rank r, α, target the attention q/v projections), applied to ViT blocks. |
| *DeeCLIP: A Robust and Generalizable Transformer-Based Framework for Detecting AI-Generated Images*, 2025 | https://arxiv.org/abs/2504.19876 | LoRA inside CLIP ViT-L/14 plus shallow/deep feature fusion. Supports LoRA on CLIP for this task, and the idea of probing intermediate layers. |
| *Mixture of Low-rank Experts for Transferable AI-Generated Image Detection*, 2024 | https://arxiv.org/abs/2404.04883 | An earlier low-rank adaptation of CLIP for cross-generator transfer. An alternative design if plain LoRA overfits. |
| *Boosting Robust AIGI Detection with LoRA-based Pairwise Training*, 2026 (NTIRE 2026 entry) | https://arxiv.org/abs/2604.12307 | LoRA plus degradation-aware training performing well in a robustness challenge. Motivates training with JPEG/resize augmentation on both classes. |
| Oquab et al. *DINOv2*, 2023 · Darcet et al. *Vision Transformers Need Registers*, 2023 | https://arxiv.org/abs/2304.07193 · https://arxiv.org/abs/2309.16588 | Background for DINO-family backbones. Used as a fallback if DINOv3 gating is a problem. |

Counterpoint to keep in mind: Zhou et al. (2602.01738, above) argue that frozen features are already strong. Our Track 1 vs Track 2 comparison on the **same backbone** tests whether LoRA helps or hurts on *unseen* generators.

## Track 3: ResNet-50 from scratch with blur+JPEG augmentation

| Paper | Link | What we take from it |
|---|---|---|
| Wang, Wang, Zhang, Owens, Efros. *CNN-generated images are surprisingly easy to spot… for now* (CNNDetection), CVPR 2020 | https://arxiv.org/abs/1912.11035 · https://peterwang512.github.io/CNNDetection/ | The recipe: ResNet-50 binary classifier, Gaussian-blur and JPEG augmentation, which this paper reports improves generalisation. Exact probabilities and σ ranges come from their code, not from memory. |
| He et al. *Deep Residual Learning for Image Recognition*, 2015 | https://arxiv.org/abs/1512.03385 | Architecture. We train from scratch (no ImageNet weights) to keep the "no pretraining" comparison clean with Track 5. |

## Track 4: NPR (up-sampling artefacts)

| Paper | Link | What we take from it |
|---|---|---|
| Tan et al. *Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection*, CVPR 2024 | https://arxiv.org/abs/2312.10461 · code https://github.com/chuangchuangtan/NPR-DeepfakeDetection | Neighbouring Pixel Relationships: the residual between an image and its down-then-up-sampled version, fed to a small ResNet. We re-implement it and **check it against the official code** with a parity test on a few images. |

## Track 5: Vision Transformer from scratch

| Paper | Link | What we take from it |
|---|---|---|
| Dosovitskiy et al. *An Image is Worth 16x16 Words* (ViT), ICLR 2021 | https://arxiv.org/abs/2010.11929 | Architecture, patch embedding, learned position embeddings. ViTs need large data without strong augmentation, which is the motivation for our scaling study. |
| Touvron et al. *Training data-efficient image transformers & distillation through attention* (DeiT), ICML 2021 | https://arxiv.org/abs/2012.12877 | The training recipe: AdamW, warmup + cosine, label smoothing, stochastic depth, strong augmentation. ViT-Ti/S configs. We **drop** colour/geometric augmentations that destroy forensic traces (RandAugment colour ops, resize-based RandomResizedCrop). |
| Touvron et al. *DeiT III: Revenge of the ViT*, 2022 | https://arxiv.org/abs/2204.07118 | Simpler augmentation ("3-Augment") and simple random crops. Closer to what forensics allows. |
| Steiner et al. *How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers*, TMLR 2022 | https://arxiv.org/abs/2106.10270 | The trade-off between data size and augmentation/regularisation. Shapes how we tune regularisation per data scale (10k vs 200k). |
| Beyer, Zhai, Kolesnikov. *Better plain ViT baselines for ImageNet-1k*, 2022 | https://arxiv.org/abs/2205.01580 | Fixed 2D sin-cos position embeddings and global average pooling instead of a CLS token. Our cheap position-embedding ablation. |
| Lee, Lee, Song. *Vision Transformer for Small-Size Datasets* (SPT + LSA), 2021 | https://arxiv.org/abs/2112.13492 | Locality-inducing tokenisation for small data. An optional fix if ViT at 10k–50k fails to train. |
| Huang et al. *Deep Networks with Stochastic Depth*, 2016 (ECCV) | https://arxiv.org/abs/1603.09382 | The regulariser in the DeiT recipe. |
| He et al. *Masked Autoencoders Are Scalable Vision Learners* (MAE), CVPR 2022 | https://arxiv.org/abs/2111.06377 | Stretch goal (A40 only): self-supervised pretraining on unlabeled real+fake crops, then fine-tuning. |
| Fridrich & Kodovský. *Rich Models for Steganalysis of Digital Images* (SRM), IEEE TIFS 2012. **UNVERIFIED** (no primary page retrieved) | n/a | Fixed high-pass residual filters, used for the **noise-residual input stem** ablation. |
| Bayar & Stamm. Constrained convolutional layer for manipulation detection, IH&MMSec 2016. **UNVERIFIED** (no primary page retrieved; secondary descriptions in https://arxiv.org/abs/2112.08935) | n/a | A learnable high-pass stem (centre weight −1, others sum to 1). Optional variant of the residual stem. |

## Track 6 (optional): training-free reconstruction error

| Paper | Link | What we take from it |
|---|---|---|
| Ricker, Lukovnikov, Fischer. *AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error*, CVPR 2024 | https://arxiv.org/abs/2401.17879 · https://openaccess.thecvf.com/content/CVPR2024/html/Ricker_AEROBLADE_Training-Free_Detection_of_Latent_Diffusion_Images_Using_Autoencoder_Reconstruction_CVPR_2024_paper.html | Score = LPIPS-based reconstruction distance through LDM autoencoders (minimum over several VAEs). Expected to be strong on LDM fakes and weak on non-LDM generators and after JPEG. A useful contrast. |
| Rajan, Ojha, Schloesser, Lee. *Aligned Datasets Improve Detection of Latent Diffusion-Generated Images*, ICLR 2025 | https://arxiv.org/abs/2410.11835 | Training on real images vs their VAE reconstructions removes content/size shortcuts. A cheap extra-fake idea and a warning about shortcuts. |

## Data quality, shortcuts and robustness (applies to all tracks)

| Paper | Link | What we take from it |
|---|---|---|
| Grommelt, Weiss, Pfreundt, Keuper. *Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets*, ECCV Workshops 2024 | https://arxiv.org/abs/2403.17608 | GenImage contains JPEG-format and image-size biases that detectors exploit. This is the direct motivation for our **shortcut audit** and **identical compression handling**. |
| Guillaro et al. *A Bias-Free Training Paradigm for More General AI-generated Image Detection* (B-Free), CVPR 2025 | https://arxiv.org/abs/2412.17671 · https://grip-unina.github.io/B-Free | Semantic/content bias between classes, and aligned real/fake training data. Motivates the CLIP content-bias check and the caption-matched 2026 set. |
| Park & Owens. *Community Forensics*, CVPR 2025 | https://arxiv.org/abs/2411.04125 | Generator diversity (thousands of models) improves generalisation. Our main training source. |
| Pellegrini et al. *AI-GenBench*, IJCNN 2025 | https://arxiv.org/abs/2504.20865 | A temporal protocol: train on older generators, test on newer. We use its newest windows as a "future" test. |
| Yan et al. *A Sanity Check for AI-generated Image Detection* (AIDE, Chameleon), ICLR 2025 | https://arxiv.org/abs/2406.19435 | A hard test set (images humans misclassified). An optional test source. |
| Gushchin et al. *NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild*, CVPRW 2026 | https://arxiv.org/abs/2604.11487 | Its 36 degradation types and top-method trends (foundation models, ensembles, degradation training) informed our degradation list. |
| Konstantinidou et al. *Navigating the Challenges of AI-Generated Image Detection in the Wild: What Truly Matters?*, 2025 | https://arxiv.org/abs/2507.10236 | Backbone, training data, cropping and augmentation as the factors that matter in the wild. Supports crops over resizing. |
| *Is Artificial Intelligence Generated Image Detection a Solved Problem?* (AIGIBench), 2025 | https://arxiv.org/abs/2505.12335 | Reports large drops under JPEG/noise. Context for the robustness section. |
| RRDataset (social-media transmission + re-digitisation benchmark), 2025. Exact title **UNVERIFIED** | https://arxiv.org/abs/2509.09172 | Multi-round transmission and re-digitisation as test conditions. Inspiration for the screenshot chain. |
| Zhu et al. *GenImage*, NeurIPS 2023 D&B | https://arxiv.org/abs/2306.08571 | Training subset and Midjourney validation. Known bias handled by the audit. |

## Video

| Paper | Link | What we take from it |
|---|---|---|
| *DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark*, 2024 | https://arxiv.org/abs/2405.19707 · https://github.com/chenhaoxing/DeMamba | GenVideo data, a cross-generator video protocol, and a degraded-video evaluation. Its temporal module is out of scope; we use frame-level detection + pooling. |
| Ni et al. *GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection*, AAAI 2026 | https://arxiv.org/abs/2501.11340 · https://genvidbench.github.io/ | Prompt-paired real/fake videos (content is not a shortcut), and cross-source/cross-generator splits. The alternative video source. |
