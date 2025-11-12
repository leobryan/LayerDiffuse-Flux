# Inference Guide

本指南说明如何使用训练好的模型进行推理。

## 模型格式说明

根据你训练保存的模型格式不同，使用不同的推理脚本：

### 1. 旧格式 LoRA (已训练的模型)
**特征：** checkpoint目录包含 `adapter_model.safetensors` 或 `model.safetensors`

**使用脚本：** `inference_old_lora.py`

**示例：**
```bash
python inference_old_lora.py \
    --base_model "./models/flux_merged_base" \
    --lora_checkpoint "./style_lora_output/checkpoint-1000" \
    --trans_vae "./models/TransparentVAE.pth" \
    --prompt "glass bottle, high quality" \
    --width 1024 \
    --height 1024 \
    --steps 50 \
    --guidance 3.5 \
    --seed 42
```

或使用便捷脚本：
```bash
# 编辑 test_old_lora.sh 设置路径和参数
bash test_old_lora.sh
```

### 2. 新格式 LoRA (使用修复后的训练脚本)
**特征：** checkpoint目录包含 `pytorch_lora_weights.safetensors`

**使用脚本：** `demo_t2i.py` (原repo脚本)

**示例：**
```bash
python demo_t2i.py \
    --ckpt_path "./models/flux_merged_base" \
    --lora_weights "./style_lora_output/final_model" \
    --trans_vae "./models/TransparentVAE.pth" \
    --prompt "glass bottle, high quality" \
    --width 1024 \
    --height 1024 \
    --steps 50 \
    --guidance 3.5 \
    --seed 42
```

### 3. Full Model (整个 transformer)
**特征：** checkpoint目录包含 `config.json` 和 `diffusion_pytorch_model.safetensors`

**使用脚本：** `inference_full_model.py`

**示例：**
```bash
python inference_full_model.py \
    --base_model "./models/flux_merged_base" \
    --finetuned_transformer "./style_lora_output/checkpoint-1000" \
    --trans_vae "./models/TransparentVAE.pth" \
    --prompt "glass bottle, high quality" \
    --width 1024 \
    --height 1024 \
    --steps 50 \
    --guidance 3.5 \
    --seed 42
```

或使用便捷脚本：
```bash
# 编辑 inference_example.sh 设置路径和参数
bash inference_example.sh
```

## 如何确定模型格式

检查你的checkpoint目录：

```bash
ls -la ./style_lora_output/checkpoint-1000/
```

根据文件判断：

| 文件名 | 格式类型 | 使用脚本 |
|--------|---------|---------|
| `adapter_model.safetensors` | 旧格式 LoRA (PEFT) | `inference_old_lora.py` |
| `model.safetensors` | 旧格式 LoRA (PEFT) | `inference_old_lora.py` |
| `pytorch_lora_weights.safetensors` | 新格式 LoRA (diffusers) | `demo_t2i.py` |
| `diffusion_pytorch_model.safetensors` | Full Model | `inference_full_model.py` |

## 参数说明

### 必需参数
- `--base_model`: 基础Flux模型路径（通常是merged base model）
- `--lora_checkpoint` / `--lora_weights` / `--finetuned_transformer`: 训练好的模型路径
- `--trans_vae`: TransparentVAE权重路径
- `--prompt`: 文本提示词

### 可选参数
- `--width`, `--height`: 图像尺寸（必须是128的倍数，默认1024）
- `--steps`: 推理步数（默认50）
- `--guidance`: 引导系数（默认3.5）
- `--seed`: 随机种子（默认42）
- `--output_dir`: 输出目录（默认 `./inference_outputs`）
- `--output_name`: 输出文件名（不指定则使用时间戳）

## 常见问题

### Q1: KeyError 加载LoRA失败
**原因：** LoRA格式不匹配

**解决：**
- 如果有 `adapter_model.safetensors`，使用 `inference_old_lora.py`
- 如果重新训练，使用修复后的训练脚本（会生成正确格式）

### Q2: 图像尺寸错误
**错误：** `Width and height must be divisible by 128`

**解决：** 使用128的倍数，例如：
- 正方形：1024x1024
- 竖图：768x1280
- 横图：1280x768

### Q3: 内存不足
**解决：**
1. 减小图像尺寸
2. 减少推理步数
3. 使用 `--dtype float16` 代替 bfloat16

## 批量生成示例

创建一个批量生成脚本：

```bash
#!/bin/bash
# batch_inference.sh

BASE_MODEL="./models/flux_merged_base"
LORA_CHECKPOINT="./style_lora_output/checkpoint-1000"
TRANS_VAE="./models/TransparentVAE.pth"
OUTPUT_DIR="./batch_outputs"

# 多个提示词
PROMPTS=(
    "glass bottle, high quality"
    "crystal vase, artistic"
    "transparent sphere, beautiful"
    "ice sculpture, detailed"
)

# 多个种子
for seed in 42 123 456 789; do
    for prompt in "${PROMPTS[@]}"; do
        echo "Generating: $prompt (seed: $seed)"
        python inference_old_lora.py \
            --base_model "$BASE_MODEL" \
            --lora_checkpoint "$LORA_CHECKPOINT" \
            --trans_vae "$TRANS_VAE" \
            --prompt "$prompt" \
            --seed $seed \
            --output_dir "$OUTPUT_DIR"
    done
done
```

## 输出格式

生成的图像为PNG格式，包含alpha通道（透明度）。

可以在支持透明度的软件中查看：
- Photoshop
- GIMP
- Preview (macOS)
- 现代浏览器

## 性能优化

### GPU内存优化
```python
# 在脚本中添加
torch.cuda.empty_cache()
```

### 批量推理
如果要生成多张图片，可以一次性传入多个prompt：
```python
prompts = ["prompt1", "prompt2", "prompt3"]
# 修改脚本支持批量处理
```

### FP16精度
使用 `--dtype float16` 可以节省约50%的显存，但可能略微降低质量。
