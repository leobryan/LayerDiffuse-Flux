# 显存优化指南 / VRAM Optimization Guide

如果你遇到显存不足（OOM）问题，以下是几种优化方案。

## 问题诊断

### 显存占用过高的原因

1. **基础模型占用**: Flux 模型本身约 12-15GB
2. **VAE 占用**: TransparentVAE 约 2-3GB
3. **Text Encoders**: CLIP + T5 约 5-6GB
4. **训练激活值**: 前向和反向传播的中间结果 8-15GB
5. **优化器状态**: AdamW 优化器 额外 2-4GB
6. **批次数据**: 图像和 latent 数据 2-4GB

**总计**: 标准配置约需要 30-40GB 显存

---

## 优化方案

### 方案1: 启用 Gradient Checkpointing（推荐）

**效果**: 减少 30-40% 显存，轻微降低速度（~10%）

Gradient checkpointing 已默认启用，如需禁用：

```bash
# 禁用 gradient checkpointing（不推荐）
--no-gradient-checkpointing
```

### 方案2: CPU Offloading

**效果**: 减少 50-60% 显存，但训练速度降低 40-50%

```bash
accelerate launch --mixed_precision="bf16" train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --output_dir "./output/style_lora" \
    --enable_cpu_offload \
    --gradient_checkpointing \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    ... (其他参数)
```

**工作原理**:
- VAE 和 Text Encoders 保留在 CPU 内存
- 仅在需要时临时移到 GPU
- 使用后立即释放 GPU 显存

### 方案3: 降低图像分辨率

**效果**: 显存占用与分辨率平方成正比

| 分辨率 | 显存占用 | 质量 |
|--------|---------|------|
| 1024x1024 | ~35GB | 最佳 |
| 768x768 | ~20GB | 良好 |
| 576x576 | ~12GB | 可接受 |
| 512x512 | ~10GB | 基础 |

⚠️ **重要限制**: 所有尺寸必须是 **128 的倍数**（Flux VAE 16倍 × TransparentVAE 8倍下采样）

**竖屏/横屏优化** (像素数相同，显存占用类似):

| 配置 | 分辨率 | 显存 |
|------|--------|------|
| 标准竖屏 | 1280x768 | ~33GB |
| 中等竖屏 | 1024x640 | ~25GB |
| 低显存竖屏 | 768x512 | ~18GB |
| 最小竖屏 | 640x512 | ~16GB |

```bash
# 使用较低分辨率（必须是 128 的倍数）
--height 1024 --width 640  # 竖屏，中等显存
# 或
--height 768 --width 512   # 竖屏，低显存
# 或
--height 512 --width 512   # 方形，最低显存
```

### 方案4: 降低 LoRA Rank

**效果**: 减少可训练参数数量

```bash
--lora_rank 8   # 从 16 降低到 8，减少 ~50% LoRA 参数
```

| Rank | 参数量 | 显存节省 | 表达能力 |
|------|--------|---------|---------|
| 32 | 最多 | 0% (基准) | 最强 |
| 16 | 中等 | ~30% | 推荐 |
| 8 | 较少 | ~60% | 良好 |
| 4 | 最少 | ~75% | 基础 |

### 方案5: 调整 Batch Size 和梯度累积

保持有效 batch size 不变，但减少单次显存占用：

```bash
# 原配置: batch_size=2, accumulation=4, 有效batch=8
--batch_size 2 --gradient_accumulation_steps 4

# 低显存配置: 有效batch仍为8
--batch_size 1 --gradient_accumulation_steps 8
```

### 方案6: 使用 8-bit 优化器

**效果**: 减少优化器状态显存 ~60%

```bash
pip install bitsandbytes
```

然后在训练脚本中修改优化器（需要手动编辑 `train_style_lora.py`）:

```python
# 将
optimizer = torch.optim.AdamW(...)

# 改为
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(...)
```

---

## 组合优化方案

### 40GB 显卡（A100, A6000）- 标准配置

```bash
accelerate launch --mixed_precision="bf16" train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --resolution 1024 \
    --aspect_ratio_type "square" \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --lora_rank 16
```

**预期显存**: ~32-38GB

### 24GB 显卡（RTX 3090/4090, A5000）- 优化配置

```bash
accelerate launch --mixed_precision="bf16" train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --height 1024 \
    --width 640 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --enable_cpu_offload \
    --lora_rank 8
```

**预期显存**: ~20-24GB

### 16GB 显卡（RTX 4080, V100 16GB）- 极限优化

```bash
accelerate launch --mixed_precision="bf16" train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --height 512 \
    --width 512 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --enable_cpu_offload \
    --lora_rank 4
```

**预期显存**: ~14-16GB
**⚠️ 警告**: 训练速度会显著降低（~3-4倍慢）

---

## 监控显存使用

### 实时监控

```bash
# 终端1: 启动训练
accelerate launch ... train_style_lora.py ...

# 终端2: 监控显存
watch -n 1 nvidia-smi
```

### Python 代码内监控

在训练脚本中添加（用于调试）:

```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
```

---

## 故障排查

### 问题: 显存在训练开始前就满了

**原因**: 模型加载时占用过多
**解决**: 使用 `--enable_cpu_offload`

### 问题: 第一个 batch 成功，第二个 batch OOM

**原因**: 显存未正确释放
**解决**:
1. 检查是否有 tensor 泄漏
2. 确保 `torch.cuda.empty_cache()` 被调用
3. 减小 batch size

### 问题: CPU offload 后训练非常慢

**原因**: CPU-GPU 数据传输是瓶颈
**建议**:
1. 使用更低分辨率而不是 CPU offload
2. 确保使用 NVMe SSD
3. 增加系统内存（推荐 64GB+）

### 问题: 启用 gradient checkpointing 后训练变慢

**预期行为**: Gradient checkpointing 会降低 10-20% 速度
**权衡**: 显存节省 > 速度损失

---

## 最佳实践

1. **优先级排序**:
   ```
   1. Gradient Checkpointing (几乎无损)
   2. 降低分辨率 (中等权衡)
   3. 降低 LoRA Rank (小影响)
   4. CPU Offload (大速度损失)
   ```

2. **测试策略**:
   - 先用最小配置测试 1-2 个 epoch
   - 确认训练可以运行
   - 逐步增加分辨率/rank

3. **监控指标**:
   - 显存峰值使用量
   - 训练速度（steps/sec）
   - 损失曲线是否正常

4. **数据准备**:
   - 预先处理图像到目标分辨率
   - 避免在训练时实时缩放大图

---

## 参考配置表

| 显卡 | 分辨率 | Rank | CPU Offload | 预期显存 | 速度 |
|------|--------|------|-------------|---------|------|
| A100 40GB | 1024x1024 | 16 | ❌ | 35GB | 1.0x |
| A100 40GB | 1280x768 | 16 | ❌ | 33GB | 1.0x |
| RTX 4090 24GB | 1024x640 | 8 | ✅ | 22GB | 0.5x |
| RTX 3090 24GB | 768x512 | 8 | ✅ | 18GB | 0.5x |
| V100 16GB | 512x512 | 4 | ✅ | 15GB | 0.3x |

**速度**: 相对于 A100 40GB 标准配置的训练速度

---

如有其他显存问题，请检查：
1. 是否有其他进程占用 GPU
2. 是否使用了正确的 dtype (bf16 或 fp16)
3. 驱动和 CUDA 版本是否匹配
