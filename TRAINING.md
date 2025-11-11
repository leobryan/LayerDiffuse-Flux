# 风格微调训练指南 / Style Fine-tuning Training Guide

本指南介绍如何基于 LayerDiffuse-Flux 的 transparent VAE 和 LoRA 进行风格微调训练。

This guide explains how to perform style fine-tuning on top of LayerDiffuse-Flux's transparent VAE and LoRA.

---

## 📋 目录 / Table of Contents

1. [训练流程概述](#训练流程概述)
2. [环境准备](#环境准备)
3. [数据准备](#数据准备)
4. [步骤1: 合并LoRA到基础模型](#步骤1-合并lora到基础模型)
5. [步骤2: 训练新的风格LoRA](#步骤2-训练新的风格lora)
6. [步骤3: 使用训练好的LoRA](#步骤3-使用训练好的lora)
7. [训练参数调优](#训练参数调优)
8. [常见问题](#常见问题)

---

## 训练流程概述

### 整体思路

```
原始 Flux Model + layerlora.safetensors + TransparentVAE
                    ↓
            【步骤1: 合并LoRA】
                    ↓
        合并后的基础模型 (Merged Base Model)
                    ↓
        【步骤2: 在风格数据集上训练新LoRA】
                    ↓
            风格 LoRA (Style LoRA)
                    ↓
        【步骤3: 推理生成带风格的透明图像】
```

### 关键点

1. **保留透明度能力**: 通过合并原始的 `layerlora.safetensors` 到基础模型，确保模型保持生成透明图像的能力
2. **风格迁移**: 在合并后的模型上训练新的 LoRA，学习特定的艺术风格
3. **两阶段训练**: 先固化透明度能力，再叠加风格能力，避免能力冲突

---

## 环境准备

### 1. 安装依赖

确保已安装所有必要的 Python 包:

```bash
pip install -r requirements.txt
```

如果需要训练，还需要安装以下额外依赖:

```bash
pip install accelerate tensorboard
```

### 2. 下载预训练模型

```bash
# 下载 TransparentVAE 和 layerlora
huggingface-cli download --resume-download --local-dir ./models RedAIGC/Flux-version-LayerDiffuse

# 确认文件存在
ls ./models/
# 应该看到:
#   - TransparentVAE.pth
#   - layerlora.safetensors
```

### 3. 硬件要求

- **最低配置**: NVIDIA GPU with 24GB VRAM (RTX 3090/4090, A5000)
- **推荐配置**: NVIDIA GPU with 40GB+ VRAM (A100, A6000)
- **训练时间**: 根据数据集大小，单卡训练约 4-12 小时

---

## 数据准备

### 数据格式要求

训练数据应该是带 alpha 通道的 PNG 图像，以及对应的文本描述。

### 目录结构

**方式1: 使用 metadata.jsonl** (推荐)

```
training_data/
├── images/
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── metadata.jsonl
```

`metadata.jsonl` 格式 (每行一个 JSON 对象):

```jsonl
{"file_name": "image001.png", "text": "a transparent glass bottle in anime style"}
{"file_name": "image002.png", "text": "a crystal flower with watercolor effects"}
```

**方式2: 使用独立的文本文件**

```
training_data/
└── images/
    ├── image001.png
    ├── image001.txt
    ├── image002.png
    ├── image002.txt
    └── ...
```

每个 `.txt` 文件包含对应图像的描述文本。

### 数据集质量建议

1. **图像数量**:
   - 最少: 50-100 张高质量图像
   - 推荐: 200-500 张
   - 更多图像可以提高泛化能力

2. **图像质量**:
   - 分辨率: 1024x1024 (或可被调整到此分辨率)
   - 格式: PNG with alpha channel
   - Alpha 通道清晰，边缘平滑

3. **文本描述**:
   - 准确描述图像内容和风格
   - 包含关键风格词汇 (如 "anime style", "watercolor", "oil painting" 等)
   - 长度: 5-20 个单词

### 测试数据集

可以使用仓库中的示例图像测试:

```bash
# 创建测试数据集
mkdir -p test_training_data/images
cp imgs/causal_cut.png test_training_data/images/
echo "a beautiful transparent glass object" > test_training_data/images/causal_cut.txt

# 测试数据加载器
python dataset.py --data_dir test_training_data --batch_size 1
```

---

## 步骤1: 合并LoRA到基础模型

### 原理

将 `layerlora.safetensors` 的权重合并到 Flux 基础模型中，创建一个新的"基础模型"，该模型原生支持生成透明图像。

### 执行命令

```bash
python merge_lora.py \
    --base_model "black-forest-labs/FLUX.1-dev" \
    --lora_weights "./models/layerlora.safetensors" \
    --output_dir "./models/flux_merged_base" \
    --lora_scale 1.0 \
    --dtype "bfloat16"
```

### 参数说明

- `--base_model`: Flux 基础模型路径 (可以是本地路径或 HuggingFace 模型 ID)
- `--lora_weights`: layerlora 权重文件路径
- `--output_dir`: 合并后模型的保存路径
- `--lora_scale`: LoRA 合并比例 (0.0-1.0)
  - `1.0`: 完全合并 (推荐)
  - `0.8`: 80% 合并，保留部分原始特性
- `--dtype`: 模型精度 (`bfloat16` 推荐，节省显存)

### 输出

合并后的模型将保存到 `./models/flux_merged_base/`，包含完整的模型权重。

### 验证

可以使用合并后的模型进行测试:

```bash
python demo_t2i.py \
    --ckpt_path "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --prompt "a transparent glass bottle" \
    --output_dir "./merge_test_outputs"
```

---

## 步骤2: 训练新的风格LoRA

### 原理

在合并后的基础模型上，使用你的风格数据集训练一个新的 LoRA 适配器。新 LoRA 将学习特定的艺术风格，同时保留透明度生成能力。

### 单 GPU 训练

```bash
export CUDA_VISIBLE_DEVICES=0

accelerate launch --mixed_precision="bf16" train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --output_dir "./output/style_lora_$(date +%Y%m%d_%H%M%S)" \
    --resolution 1024 \
    --center_crop \
    --random_flip \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 100 \
    --learning_rate 1e-4 \
    --lr_scheduler "constant_with_warmup" \
    --lr_warmup_steps 500 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --mixed_precision "bf16" \
    --checkpointing_steps 500 \
    --seed 42
```

### 多 GPU 训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate config  # 配置多GPU训练

accelerate launch --mixed_precision="bf16" --num_processes=4 train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --output_dir "./output/style_lora_multi_gpu" \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    ... (其他参数同上)
```

### 关键参数说明

#### 数据参数
- `--data_dir`: 训练数据目录
- `--resolution`: 训练图像分辨率/基础尺寸 (推荐 1024)
- `--aspect_ratio_type`: 长宽比类型
  - `square`: 1:1 方形 (1024x1024) - 默认
  - `portrait`: 9:16 竖屏 (768x1360)
  - `landscape`: 16:9 横屏 (1024x576)
  - `auto`: 保持原始长宽比
- `--height` / `--width`: 显式指定高度和宽度 (覆盖 resolution 和 aspect_ratio_type)
- `--center_crop`: 是否中心裁剪
- `--random_flip`: 是否随机水平翻转 (数据增强)

#### 训练参数
- `--batch_size`: 每个 GPU 的 batch size
- `--gradient_accumulation_steps`: 梯度累积步数
  - 实际 batch size = `batch_size × gradient_accumulation_steps × num_gpus`
  - 推荐设置为 4-8 以获得稳定训练
- `--num_train_epochs`: 训练轮数
- `--max_train_steps`: 最大训练步数 (可选，覆盖 epochs)

#### 学习率参数
- `--learning_rate`: 学习率 (推荐 1e-4 到 5e-4)
- `--lr_scheduler`: 学习率调度器
  - `constant_with_warmup`: 固定学习率 + warmup (推荐)
  - `cosine`: 余弦退火
  - `linear`: 线性衰减
- `--lr_warmup_steps`: Warmup 步数 (推荐 500-1000)

#### LoRA 配置
- `--lora_rank`: LoRA 秩 (推荐 8-32)
  - 更高的秩 = 更强的表达能力，但更容易过拟合
  - 推荐: 小数据集用 8-16，大数据集用 16-32
- `--lora_alpha`: LoRA 缩放因子 (通常设置为 rank 的值)
- `--lora_dropout`: LoRA dropout (0.0-0.1，防止过拟合)
- `--lora_target_modules`: 要应用 LoRA 的模块
  - 默认: `["to_q", "to_k", "to_v", "to_out.0"]` (注意力层)

#### 其他参数
- `--mixed_precision`: 混合精度训练 (`bf16` 推荐)
- `--checkpointing_steps`: 保存检查点的间隔步数
- `--guidance_scale`: 无分类器引导尺度 (推荐 3.5)
- `--use_offset`: 是否使用 TransparentVAE 的 offset 编码 (必须开启)

### 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir ./output/style_lora_XXXXXX/logs

# 在浏览器中打开: http://localhost:6006
```

### 训练非方形图像 (9:16 / 16:9)

如果你的训练数据主要是长图像（如竖屏 9:16 或横屏 16:9），需要特别配置：

#### 方式1: 使用预设长宽比

**训练竖屏图像 (9:16 Portrait)**

```bash
accelerate launch --mixed_precision="bf16" train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --output_dir "./output/style_lora_portrait" \
    --aspect_ratio_type "portrait" \
    --resolution 1024 \
    --center_crop \
    --random_flip \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 100 \
    --learning_rate 1e-4
```

**训练横屏图像 (16:9 Landscape)**

```bash
accelerate launch --mixed_precision="bf16" train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --output_dir "./output/style_lora_landscape" \
    --aspect_ratio_type "landscape" \
    --resolution 1024 \
    --center_crop \
    --random_flip \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 100 \
    --learning_rate 1e-4
```

#### 方式2: 自定义尺寸

如果需要精确控制尺寸（例如 576x1024 的竖屏图）：

```bash
accelerate launch --mixed_precision="bf16" train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --output_dir "./output/style_lora_custom" \
    --height 1024 \
    --width 576 \
    --center_crop \
    --random_flip \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 100 \
    --learning_rate 1e-4
```

#### 方式3: 保持原始长宽比

如果数据集包含多种长宽比，可以使用 `auto` 模式：

```bash
accelerate launch --mixed_precision="bf16" train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --output_dir "./output/style_lora_auto" \
    --aspect_ratio_type "auto" \
    --resolution 1024 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 100 \
    --learning_rate 1e-4
```

⚠️ **注意**: `auto` 模式下每张图像会保持原始长宽比，但这可能导致批次内图像尺寸不一致，建议 `batch_size=1`。

#### 推荐尺寸

| 长宽比 | 推荐尺寸 | 说明 |
|--------|---------|------|
| 1:1 (方形) | 1024x1024 | 标准设置 |
| 9:16 (竖屏) | 768x1360 | 适合手机竖屏内容 |
| 9:16 (竖屏) | 576x1024 | 显存受限时使用 |
| 16:9 (横屏) | 1024x576 | 适合横屏视频 |
| 16:9 (横屏) | 1360x768 | 更高质量横屏 |

#### 显存消耗对比

不同尺寸的显存消耗（在 batch_size=1, bf16 精度下）：

| 分辨率 | 像素数 | 预估显存 |
|--------|-------|---------|
| 1024x1024 | 1.05M | ~20GB |
| 768x1360 | 1.04M | ~20GB |
| 576x1024 | 0.59M | ~14GB |
| 1024x576 | 0.59M | ~14GB |

💡 **技巧**: 如果显存不足，可以使用 576x1024 或 1024x576，质量损失不大。

---

## 步骤3: 使用训练好的LoRA

### 文本到图像生成

```bash
python demo_t2i.py \
    --ckpt_path "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --lora_weights "./output/style_lora_XXXXXX/final_model" \
    --output_dir "./styled_outputs" \
    --prompt "a beautiful transparent glass bottle in anime style" \
    --steps 50 \
    --guidance 3.5 \
    --seed 12345
```

### 图像到图像生成

```bash
python demo_i2i.py \
    --ckpt_path "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --lora_weights "./output/style_lora_XXXXXX/final_model" \
    --input_image "./imgs/causal_cut.png" \
    --output_dir "./styled_i2i_outputs" \
    --prompt "a handsome man in anime style with colorful background" \
    --steps 50 \
    --guidance 7.0 \
    --strength 0.8
```

### 加载多个 LoRA

如果需要组合多个 LoRA (例如透明度 + 风格):

```python
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)

# 加载多个 LoRA
pipe.load_lora_weights("./path/to/style_lora", adapter_name="style")
pipe.set_adapters(["style"], adapter_weights=[1.0])

# 生成
images = pipe(prompt="...", ...).images
```

---

## 训练参数调优

### 学习率调优

| 数据集大小 | 推荐学习率 | 说明 |
|-----------|-----------|------|
| < 100 张   | 5e-5      | 小数据集需要较低学习率防止过拟合 |
| 100-500 张 | 1e-4      | 标准设置 |
| > 500 张   | 2e-4 到 5e-4 | 大数据集可以使用更高学习率 |

### LoRA Rank 调优

| Rank | 参数量 | 适用场景 |
|------|--------|---------|
| 4-8  | 最少   | 小数据集，简单风格 |
| 16   | 适中   | 通用推荐设置 |
| 32   | 较多   | 大数据集，复杂风格 |
| 64+  | 很多   | 容易过拟合，不推荐 |

### 过拟合检测

如果发现训练损失持续下降但生成质量变差，可能是过拟合:

**解决方法**:
1. 降低 `lora_rank`
2. 增加 `lora_dropout` (例如 0.05-0.1)
3. 减少训练轮数
4. 增加数据集大小
5. 使用更强的数据增强 (`--random_flip`)

### 显存优化

如果遇到 OOM (Out of Memory):

1. **降低 batch size**:
   ```bash
   --batch_size 1 --gradient_accumulation_steps 8
   ```

2. **使用梯度检查点** (添加到训练脚本):
   ```python
   pipe.transformer.enable_gradient_checkpointing()
   ```

3. **使用 8-bit 优化器**:
   ```bash
   pip install bitsandbytes
   # 修改训练脚本中的优化器为 AdamW8bit
   ```

---

## 常见问题

### Q1: 训练后的模型失去了透明度能力?

**A**: 确保:
1. 使用了合并后的基础模型 (`flux_merged_base`)
2. 训练时开启了 `--use_offset` 参数
3. TransparentVAE 路径正确
4. 训练数据包含清晰的 alpha 通道

### Q2: 训练损失不下降?

**A**: 检查:
1. 学习率是否太低 (尝试提高到 1e-4 或 2e-4)
2. 数据是否正确加载 (运行 `python dataset.py --data_dir YOUR_DATA`)
3. TransparentVAE 是否正确编码 (检查 latents 是否有 NaN)

### Q3: 生成的图像风格不明显?

**A**: 可能的原因:
1. 训练步数不够 (尝试增加到 5000-10000 步)
2. LoRA rank 太低 (尝试增加到 32)
3. 数据集风格不一致
4. 推理时 LoRA 权重太低 (尝试设置 adapter_weights 为 1.5-2.0)

### Q4: 如何在推理时控制风格强度?

**A**: 使用 LoRA 权重缩放:

```python
pipe.set_adapters(["style"], adapter_weights=[0.8])  # 80% 风格强度
```

或在加载时:

```bash
python demo_t2i.py \
    ... \
    --lora_scale 0.8  # 添加此参数
```

### Q5: 训练时间太长怎么办?

**A**: 优化建议:
1. 使用更少的验证步数 (`--validation_steps 1000`)
2. 减少 TransparentVAE 的测试时增强 (修改 `vae.py` 中的 `estimate_augmented`)
3. 使用多 GPU 训练
4. 降低分辨率 (例如 512，但会降低质量)

### Q6: 如何评估训练质量?

**A**: 多种方法:
1. 观察 TensorBoard 中的 loss 曲线
2. 定期生成验证图像
3. 使用固定种子生成对比图像
4. 人工评估生成图像的风格一致性和透明度质量

---

## 高级技巧

### 1. 混合多个风格

训练多个风格 LoRA，推理时动态混合:

```python
pipe.load_lora_weights("./style1", adapter_name="anime")
pipe.load_lora_weights("./style2", adapter_name="watercolor")

pipe.set_adapters(["anime", "watercolor"], adapter_weights=[0.7, 0.3])
```

### 2. 渐进式训练

先用低 rank 训练基础风格,再用高 rank 精调:

```bash
# 阶段1: Rank 8，快速收敛
python train_style_lora.py --lora_rank 8 --num_train_epochs 50

# 阶段2: Rank 32，精细调整
python train_style_lora.py \
    --lora_rank 32 \
    --num_train_epochs 50 \
    --resume_from_checkpoint "./output/stage1/checkpoint-XXXX"
```

### 3. 数据增强

除了 `--random_flip`，可以在 `dataset.py` 中添加:
- 颜色抖动 (Color jitter)
- 随机旋转 (小角度)
- 亮度/对比度调整

---

## 引用与致谢

本训练流程基于以下项目:
- [LayerDiffuse](https://github.com/layerdiffusion/LayerDiffuse)
- [Flux](https://github.com/black-forest-labs/flux)
- [Diffusers](https://github.com/huggingface/diffusers)
- [PEFT](https://github.com/huggingface/peft)

---

## 许可证

训练代码遵循与原仓库相同的许可证。训练得到的模型权重许可证取决于你的训练数据和基础模型。

---

**祝训练顺利!** 🎨✨

如有问题,请在 GitHub Issues 中提出。
