# NaN Loss 故障排查指南

当训练时出现 `loss=nan`，这通常表示数值不稳定。以下是系统的诊断和解决方法。

---

## 🔍 快速诊断

### 步骤 1: 运行诊断脚本

```bash
python debug_nan_loss.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --resolution 1024 \
    --aspect_ratio_type "square"
```

这个脚本会检查：
- ✓ 训练数据是否包含 NaN/Inf
- ✓ VAE 编码是否产生有效的 latents
- ✓ 模型权重是否正常
- ✓ 前向传播是否正常

---

## 🐛 常见原因及解决方法

### 1. **学习率过高** ⭐️ 最常见原因

**症状**: 训练开始后几个 step 就出现 NaN

**解决方法**:
```bash
# 降低学习率
--learning_rate 5e-5  # 从 1e-4 降低到 5e-5
# 或更低
--learning_rate 1e-5
```

**为什么**: Flux 是一个大模型，高学习率会导致梯度爆炸。

### 2. **混合精度问题**

**症状**: 使用 bf16/fp16 时出现 NaN，但 fp32 正常

**解决方法**:
```bash
# 尝试关闭混合精度
--mixed_precision "no"

# 或使用更稳定的 fp16（带 loss scaling）
--mixed_precision "fp16"
```

**为什么**: bfloat16 的动态范围大但精度低，可能导致数值下溢。

### 3. **TransparentVAE 编码问题**

**症状**: 诊断脚本显示 latents 包含 NaN 或极大值

**解决方法**:
```bash
# 检查 alpha 值设置
# 在 lib_layerdiffuse/vae.py 中，TransparentVAE 的 alpha 默认是 300.0
# 如果过大可能导致数值溢出

# 尝试降低 alpha（需要修改代码）:
trans_vae = TransparentVAE(pipe.vae, pipe.vae.dtype, alpha=150.0)
```

### 4. **数据问题**

**症状**: 诊断脚本显示数据包含 NaN/Inf 或极端值

**可能原因**:
- 图像文件损坏
- Alpha 通道全为 0（完全透明）
- 图像尺寸不正确

**解决方法**:
```bash
# 检查数据
python dataset.py --data_dir ./training_data --batch_size 1

# 查看具体哪些图像有问题
python -c "
from PIL import Image
import numpy as np
import os

data_dir = './training_data/images'
for img_file in os.listdir(data_dir):
    if img_file.endswith('.png'):
        img = Image.open(os.path.join(data_dir, img_file)).convert('RGBA')
        arr = np.array(img)

        # 检查 NaN
        if np.isnan(arr).any():
            print(f'{img_file}: contains NaN')

        # 检查 alpha 通道
        alpha = arr[:, :, 3]
        if alpha.max() == 0:
            print(f'{img_file}: alpha channel is all zero')

        # 检查尺寸
        if img.size[0] % 128 != 0 or img.size[1] % 128 != 0:
            print(f'{img_file}: size {img.size} not divisible by 128')
"
```

### 5. **梯度爆炸**

**症状**: Loss 从正常值突然变成 NaN

**解决方法**:
```bash
# 降低梯度裁剪阈值
--max_grad_norm 0.5  # 从 1.0 降低到 0.5

# 或更激进
--max_grad_norm 0.1
```

### 6. **LoRA 初始化问题**

**症状**: 第一个 batch 就出现 NaN

**解决方法**:
```bash
# 降低 LoRA rank
--lora_rank 8  # 从 16 降低到 8

# 降低 LoRA alpha
--lora_alpha 8  # 通常等于 rank
```

### 7. **批次大小问题**

**症状**: 某些批次正常，某些批次 NaN

**解决方法**:
```bash
# 使用更小的 batch size
--batch_size 1
--gradient_accumulation_steps 8

# 确保每个批次都有效
```

---

## 🛠️ 推荐的安全配置

### 配置 A: 保守设置（最稳定）

```bash
accelerate launch --mixed_precision="no" train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --resolution 1024 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --lr_scheduler "constant" \
    --lora_rank 8 \
    --lora_alpha 8 \
    --max_grad_norm 0.5 \
    --gradient_checkpointing
```

**特点**:
- ✓ 不使用混合精度（最稳定但最慢）
- ✓ 非常低的学习率
- ✓ 小的 LoRA rank
- ✓ 严格的梯度裁剪

### 配置 B: 平衡设置（推荐）

```bash
accelerate launch --mixed_precision="bf16" train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --resolution 1024 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --lr_scheduler "constant_with_warmup" \
    --lr_warmup_steps 500 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --max_grad_norm 1.0 \
    --gradient_checkpointing
```

**特点**:
- ✓ 使用 bf16 混合精度
- ✓ 适中的学习率
- ✓ 标准的 LoRA 配置
- ✓ 包含 warmup

---

## 📊 监控和调试

### 1. 监控训练指标

```bash
# 在训练日志中查看
grep "loss" training.log

# 正常的 loss 范围
# 初始: 0.01 - 0.5
# 训练中: 0.001 - 0.1
# 如果 > 1.0，可能有问题
```

### 2. 使用 TensorBoard

```bash
tensorboard --logdir ./output/style_lora_XXXXXX/logs
```

查看：
- Loss 曲线（应该平滑下降）
- Learning rate
- Gradient norm（如果突然飙升，预示 NaN）

### 3. 添加调试日志

在训练脚本中添加：

```python
# 在 compute_loss 之后
if global_step % 10 == 0:
    logger.info(f"Step {global_step}: loss={loss.item():.6f}")

    # 检查梯度
    total_norm = 0
    for p in pipe.transformer.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    logger.info(f"  Gradient norm: {total_norm:.4f}")
```

---

## 🔧 代码层面的修复

### 已实施的保护措施

训练代码已经包含以下保护：

1. **时间步裁剪**: 避免 t=0 或 t=1
   ```python
   timesteps = torch.clamp(timesteps, min=1e-7, max=1.0 - 1e-7)
   ```

2. **值裁剪**: 防止数值溢出
   ```python
   model_pred = torch.clamp(model_pred, min=-1e4, max=1e4)
   ```

3. **NaN 检测**: 每步检查并跳过 NaN batch
   ```python
   if torch.isnan(loss):
       logger.warning("NaN detected, skipping batch")
       continue
   ```

4. **梯度裁剪**: 防止梯度爆炸
   ```python
   accelerator.clip_grad_norm_(pipe.transformer.parameters(), args.max_grad_norm)
   ```

---

## 📝 故障排查清单

使用此清单逐步排查：

- [ ] 运行 `debug_nan_loss.py` 诊断脚本
- [ ] 检查数据集中是否有损坏的图像
- [ ] 确认所有图像尺寸是 128 的倍数
- [ ] 降低学习率到 1e-5 或更低
- [ ] 尝试关闭混合精度（`--mixed_precision "no"`）
- [ ] 降低 LoRA rank 到 8
- [ ] 降低梯度裁剪阈值到 0.5
- [ ] 确保使用 warmup（`--lr_warmup_steps 500`）
- [ ] 检查是否有其他进程占用 GPU 导致显存不足
- [ ] 查看 TensorBoard 确认 loss 曲线

---

## 🚨 紧急修复

如果以上都不work，尝试这个最保守的配置：

```bash
python train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --resolution 512 \
    --aspect_ratio_type "square" \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 200 \
    --learning_rate 5e-6 \
    --lr_scheduler "constant" \
    --lora_rank 4 \
    --lora_alpha 4 \
    --lora_dropout 0.1 \
    --max_grad_norm 0.1 \
    --mixed_precision "no" \
    --gradient_checkpointing \
    --seed 42
```

这个配置：
- 使用最小分辨率 512
- 极低的学习率 5e-6
- 最小的 LoRA rank 4
- 不使用混合精度
- 非常严格的梯度裁剪

虽然训练会很慢，但应该能避免 NaN。

---

## 📚 参考资料

### 学习率建议

| 模型大小 | 推荐学习率 |
|---------|-----------|
| Flux Full | 1e-6 到 5e-6 |
| Flux + LoRA | 5e-5 到 1e-4 |
| 小 LoRA (rank≤8) | 1e-4 到 5e-4 |

### LoRA 配置建议

| Rank | Alpha | Dropout | 适用场景 |
|------|-------|---------|---------|
| 4 | 4 | 0.1 | 极小数据集，最稳定 |
| 8 | 8 | 0.0 | 小数据集 (<100 图) |
| 16 | 16 | 0.0 | 标准配置 |
| 32 | 32 | 0.0 | 大数据集 (>500 图) |

---

如果问题仍然存在，请提供：
1. 完整的训练日志
2. `debug_nan_loss.py` 的输出
3. 数据集信息（图像数量、平均尺寸、格式）
4. 硬件信息（GPU 型号、显存大小）
