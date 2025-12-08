---
language:
- en
license: cc-by-4.0
size_categories:
- 10K<n<100K
task_categories:
- text-to-video
configs:
- config_name: default
  data_files:
  - split: train
    path: video_caption_train.json
---

# Usage

```
cat videos.tar.part* > videos.tar
cat masks.tar.part* > masks.tar
tar -xvf bboxes.tar
tar -xvf masks.tar
tar -xvf videos.tar
tar -xvf face_images.tar
```

For how to process your own data like ConsisID-Preview-Data dataset in the [ConsisID paper](https://huggingface.co/papers/2411.17440), please refer to [here](https://github.com/PKU-YuanGroup/ConsisID/tree/main/data_preprocess). (Support Multi-ID)

# Acknowledgement

- The current open source data is not the complete set for training ConsisID.
- The current 31.9K captions correspond to videos with a single ID, while the remaining videos have multiple IDs.
- The [data preprocess code](https://github.com/PKU-YuanGroup/ConsisID/tree/main/data_preprocess) support multi-face annotation, but the [training code](https://github.com/PKU-YuanGroup/ConsisID/blob/main/train.py) only support single-face currently.

<div align=center>
<img src="https://github.com/PKU-YuanGroup/ConsisID/blob/main/asserts/ConsisID_logo.png?raw=true" width="150px">
</div>

<h1 align="center"> <a href="https://pku-yuangroup.github.io/ConsisID">[CVPR 2025 Highlight] Identity-Preserving Text-to-Video Generation by Frequency Decomposition</a></h1>

<p style="text-align: center;">
  <a href="https://huggingface.co/spaces/BestWishYsh/ConsisID-preview-Space">🤗 Huggingface Space</a> |
  <a href="https://pku-yuangroup.github.io/ConsisID">📄 Page </a> | 
  <a href="https://github.com/PKU-YuanGroup/ConsisID">🌐 Github </a> | 
  <a href="https://huggingface.co/papers/2411.17440">📜 arxiv </a> | 
  <a href="https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data">🐳 Dataset</a>
</p>
<p align="center">
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>


## 😍 Gallery

Identity-Preserving Text-to-Video Generation. (Some best prompts [here](https://github.com/PKU-YuanGroup/ConsisID/blob/main/asserts/prompt.xlsx))
[![Demo Video of ConsisID](https://github.com/user-attachments/assets/634248f6-1b54-4963-88d6-34fa7263750b)](https://www.youtube.com/watch?v=PhlgC-bI5SQ)
or you can click <a href="https://github.com/SHYuanBest/shyuanbest_media/raw/refs/heads/main/ConsisID/showcase_videos.mp4">here</a> to watch the video.

## 🤗 Quick Start

This model supports deployment using the huggingface diffusers library. You can deploy it by following these steps.

**We recommend that you visit our [GitHub](https://github.com/PKU-YuanGroup/ConsisID) and check out the relevant prompt
optimizations and conversions to get a better experience.**

1. Install the required dependencies

```shell
# ConsisID will be merged into diffusers in the next version. So for now, you should install from source.
pip install --upgrade consisid_eva_clip pyfacer insightface facexlib transformers accelerate imageio-ffmpeg 
pip install git+https://github.com/huggingface/diffusers.git
```

2. Run the code

```python
import torch
from diffusers import ConsisIDPipeline
from diffusers.pipelines.consisid.consisid_utils import prepare_face_models, process_face_embeddings_infer
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download

snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir="BestWishYsh/ConsisID-preview")
face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std = (
    prepare_face_models("BestWishYsh/ConsisID-preview", device="cuda", dtype=torch.bfloat16)
)
pipe = ConsisIDPipeline.from_pretrained("BestWishYsh/ConsisID-preview", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# ConsisID works well with long and well-described prompts. Make sure the face in the image is clearly visible (e.g., preferably half-body or full-body).
prompt = "The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel."
image = "https://github.com/PKU-YuanGroup/ConsisID/blob/main/asserts/example_images/2.png?raw=true"

id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(
    face_helper_1,
    face_clip_model,
    face_helper_2,
    eva_transform_mean,
    eva_transform_std,
    face_main_model,
    "cuda",
    torch.bfloat16,
    image,
    is_align_face=True,
)

video = pipe(
    image=image,
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=6.0,
    use_dynamic_cfg=False,
    id_vit_hidden=id_vit_hidden,
    id_cond=id_cond,
    kps_cond=face_kps,
    generator=torch.Generator("cuda").manual_seed(42),
)
export_to_video(video.frames[0], "output.mp4", fps=8)
```

## 🛠️ Prompt Refiner

ConsisID has high requirements for prompt quality. You can use [GPT-4o](https://chatgpt.com/) to refine the input text prompt, an example is as follows (original prompt: "a man is playing guitar.")
```bash
a man is playing guitar.

Change the sentence above to something like this (add some facial changes, even if they are minor. Don't make the sentence too long): 

The video features a man standing next to an airplane, engaged in a conversation on his cell phone. he is wearing sunglasses and a black top, and he appears to be talking seriously. The airplane has a green stripe running along its side, and there is a large engine visible behind his. The man seems to be standing near the entrance of the airplane, possibly preparing to board or just having disembarked. The setting suggests that he might be at an airport or a private airfield. The overall atmosphere of the video is professional and focused, with the man's attire and the presence of the airplane indicating a business or travel context.
```

Some sample prompts are available [here](https://github.com/PKU-YuanGroup/ConsisID/blob/main/asserts/prompt.xlsx).

### 💡 GPU Memory Optimization

ConsisID requires about 44 GB of GPU memory to decode 49 frames (6 seconds of video at 8 FPS) with output resolution 720x480 (W x H), which makes it not possible to run on consumer GPUs or free-tier T4 Colab. The following memory optimizations could be used to reduce the memory footprint. For replication, you can refer to [this](https://gist.github.com/SHYuanBest/bc4207c36f454f9e969adbb50eaf8258) script.

| Feature (overlay the previous) | Max Memory Allocated | Max Memory Reserved |
| :----------------------------- | :------------------- | :------------------ |
| -                              | 37 GB                | 44 GB               |
| enable_model_cpu_offload       | 22 GB                | 25 GB               |
| enable_sequential_cpu_offload  | 16 GB                | 22 GB               |
| vae.enable_slicing             | 16 GB                | 22 GB               |
| vae.enable_tiling              | 5 GB                 | 7 GB                |

```bash
# turn on if you don't have multiple GPUs or enough GPU memory(such as H100)
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

warning: it will cost more time in inference and may also reduce the quality.

## 🙌 Description

- **Repository:** [Code](https://github.com/PKU-YuanGroup/ConsisID), [Page](https://pku-yuangroup.github.io/ConsisID/), [Data](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data)
- **Paper:** [https://huggingface.co/papers/2411.17440](https://huggingface.co/papers/2411.17440)
- **Point of Contact:** [Shenghai Yuan](shyuan-cs@hotmail.com)
- **License:** CC-BY-4.0

## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star and citation.

```BibTeX
@inproceedings{yuan2025identity,
  title={Identity-preserving text-to-video generation by frequency decomposition},
  author={Yuan, Shenghai and Huang, Jinfa and He, Xianyi and Ge, Yunyang and Shi, Yujun and Chen, Liuhan and Luo, Jiebo and Yuan, Li},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={12978--12988},
  year={2025}
}
```