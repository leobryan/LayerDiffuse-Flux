# Pixel-Space Loss Training

## 概述

这是一个新的训练模式，在**图像空间**而不是latent空间计算loss。

## 两种训练模式对比

### 原始训练（Latent-Space Loss）
**文件：** `train_style_lora.py`

**流程：**
```
1. Encode图像 → latent
2. 添加噪声 → noisy latent
3. 模型预测 → pred_noise
4. Loss = ||pred_noise - target_noise||²  ← 在latent空间
```

**优点：**
- ✅ 训练速度快（不需要decode）
- ✅ 显存占用小
- ✅ 标准扩散模型训练方式

**缺点：**
- ❌ Loss在latent空间，不直观
- ❌ 可能不完全对应图像质量
- ❌ 难以直接优化特定图像特征

### 新模式（Pixel-Space Loss）
**文件：** `train_style_lora_pixel_loss.py`

**流程：**
```
1. Encode图像 → initial latent
2. 完整生成流程（多步denoising）→ final latent
3. Decode → pred_image
4. Loss = ||pred_image - target_image||²  ← 在图像空间
```

**优点：**
- ✅ 直接优化图像质量
- ✅ Loss更直观（直接看图像差异）
- ✅ 可以分别优化RGB和Alpha
- ✅ 更容易控制生成质量

**缺点：**
- ❌ 训练速度慢（每步需要完整生成+decode）
- ❌ 显存占用大
- ❌ 需要调整inference steps和batch size

## 使用方法

### 1. 基本训练

```bash
bash train_pixel_loss.example.sh
```

或者：

```bash
accelerate launch --mixed_precision=bf16 train_style_lora_pixel_loss.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --resolution 1024 \
    --aspect_ratio_type square \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 3000 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --rgb_loss_weight 1.0 \
    --alpha_loss_weight 1.0 \
    --loss_type l2 \
    --num_inference_steps 28
```

### 2. 关键参数

#### Loss相关
- `--rgb_loss_weight`: RGB通道loss权重（默认1.0）
- `--alpha_loss_weight`: Alpha通道loss权重（默认1.0）
- `--loss_type`: Loss函数类型
  - `l1`: L1 loss (MAE) - 对outlier不敏感
  - `l2`: L2 loss (MSE) - 标准选择
  - `huber`: Huber loss - L1和L2的折中

#### 生成相关
- `--num_inference_steps`: 每个训练step的生成步数
  - 28: 快速训练，质量略低
  - 50: 平衡速度和质量（推荐）
  - 100: 高质量，但训练很慢
- `--guidance_scale`: CFG强度（默认3.5）

#### 内存优化
- `--train_batch_size`: **建议保持为1**
- `--gradient_accumulation_steps`: 增大来补偿小batch（8-16）
- `--enable_cpu_offload`: 启用CPU offload（更慢但省显存）

## 性能对比

| 指标 | Latent Loss | Pixel Loss (28 steps) | Pixel Loss (50 steps) |
|------|-------------|----------------------|----------------------|
| 速度 | 1x (baseline) | ~15x slower | ~25x slower |
| 显存 | ~20GB | ~30GB | ~30GB |
| 训练步数建议 | 5000-10000 | 2000-3000 | 1000-2000 |
| Batch Size | 4-8 | 1-2 | 1 |

## 训练建议

### 显存配置

**40GB GPU (A100/A6000):**
```bash
--train_batch_size 1 \
--gradient_accumulation_steps 8 \
--num_inference_steps 50 \
--resolution 1024
```

**24GB GPU (RTX 3090/4090):**
```bash
--train_batch_size 1 \
--gradient_accumulation_steps 4 \
--num_inference_steps 28 \
--resolution 768 \
--enable_cpu_offload  # 如果仍然OOM
```

**16GB GPU (RTX 4080):**
```bash
--train_batch_size 1 \
--gradient_accumulation_steps 2 \
--num_inference_steps 20 \
--resolution 512 \
--enable_cpu_offload
```

### Loss权重调整

**默认（平衡）：**
```bash
--rgb_loss_weight 1.0 \
--alpha_loss_weight 1.0
```

**强调透明度质量：**
```bash
--rgb_loss_weight 1.0 \
--alpha_loss_weight 2.0  # 增大alpha权重
```

**强调颜色质量：**
```bash
--rgb_loss_weight 2.0 \
--alpha_loss_weight 1.0
```

### Loss类型选择

**L2 (MSE) - 默认推荐**
```bash
--loss_type l2
```
- 标准选择
- 对大误差惩罚更重
- 适合大多数情况

**L1 (MAE)**
```bash
--loss_type l1
```
- 对outlier更robust
- 如果数据中有噪声或异常值，考虑使用
- 训练可能更稳定

**Huber Loss**
```bash
--loss_type huber
```
- L1和L2的折中
- 小误差用L2，大误差用L1
- 最robust的选择

## 训练流程

### 每个训练step的详细流程

```python
for batch in dataloader:
    # 1. Encode to latent (with TransparentVAE)
    initial_latent = trans_vae.encode(img_rgba, img_rgb, padded_rgb)

    # 2. Encode prompt
    prompt_embeds = pipe.encode_prompt(caption)

    # 3. Full generation (28-50 steps of denoising)
    for t in timesteps:
        noise_pred = transformer(latent, t, prompt_embeds)
        latent = scheduler.step(noise_pred, t, latent)

    # 4. Decode to image
    pred_image = trans_vae.decode(latent)  # (B, 4, H, W)

    # 5. Compute pixel-space loss
    loss = compute_loss(pred_image, target_image)

    # 6. Backprop
    loss.backward()
    optimizer.step()
```

## 优化建议

### 1. 减少inference steps

最大的性能瓶颈是每个训练step都要完整生成。减少`num_inference_steps`可以大幅加速：

```bash
# 快速实验
--num_inference_steps 20

# 平衡质量和速度
--num_inference_steps 28

# 高质量训练
--num_inference_steps 50
```

### 2. 使用gradient checkpointing

必须启用：
```bash
--gradient_checkpointing
```

### 3. 混合精度训练

必须使用bf16或fp16：
```bash
--mixed_precision bf16
```

### 4. CPU Offloading

如果显存不够：
```bash
--enable_cpu_offload
```

会将VAE和text encoder在不用时移到CPU，但会更慢。

## 何时使用Pixel Loss

### 推荐使用场景

✅ **精细调整风格**
- 需要精确控制图像质量
- 对透明度有特殊要求
- 需要优化特定图像特征

✅ **小数据集fine-tuning**
- 数据量较少（<100张）
- 每张图像都很重要
- 直接优化图像质量更有效

✅ **质量优先**
- 训练时间不是问题
- 追求最佳生成质量
- 有充足的GPU资源

### 不推荐使用场景

❌ **大规模训练**
- 数据集很大（>1000张）
- 训练时间有限
- 使用标准latent loss更高效

❌ **资源受限**
- GPU显存不足（<24GB）
- 训练预算有限
- 标准训练已经足够

❌ **初始探索**
- 刚开始实验
- 不确定训练效果
- 先用latent loss快速迭代

## 监控训练

训练时会输出：

```
Steps: 100%|███| 3000/3000 [2:30:00<00:00,  3.00s/it]
loss=0.0234 rgb_loss=0.0198 alpha_loss=0.0036 lr=0.0001
```

**关键指标：**
- `loss`: 总loss
- `rgb_loss`: RGB通道loss
- `alpha_loss`: Alpha通道loss
- `lr`: 当前学习率

**正常训练：**
- Loss逐渐下降
- RGB loss和alpha loss相对平衡
- 无NaN或Inf

**异常情况：**
- Loss不下降 → 学习率太小或数据有问题
- Loss爆炸 → 学习率太大
- Alpha loss远大于RGB loss → 考虑减小alpha_loss_weight

## 调试技巧

### 1. 快速验证

使用少量steps测试：
```bash
--max_train_steps 10 \
--num_inference_steps 10
```

### 2. 可视化中间结果

在训练脚本中添加：
```python
if global_step % 100 == 0:
    # 保存pred_image和target_image
    save_image(pred_image, f"debug/pred_{global_step}.png")
    save_image(target_image, f"debug/target_{global_step}.png")
```

### 3. 检查loss分布

查看RGB loss和alpha loss的比例：
```python
print(f"RGB/Alpha ratio: {rgb_loss / alpha_loss:.2f}")
```

如果比例失衡，调整权重。

## 常见问题

### Q1: 训练太慢怎么办？
**解决方案：**
1. 减少`num_inference_steps`到20-28
2. 减小`resolution`到768或512
3. 减少训练步数（pixel loss收敛更快）

### Q2: 显存不足怎么办？
**解决方案：**
1. `--train_batch_size 1`
2. `--enable_cpu_offload`
3. 减小resolution
4. 减少num_inference_steps

### Q3: Loss不下降怎么办？
**可能原因：**
- 学习率太小 → 增大到1e-3
- Inference steps太少 → 增加到50
- 数据质量问题 → 检查数据

### Q4: 生成的图像模糊？
**解决方案：**
- 增加inference steps
- 调整guidance_scale
- 检查是否正确使用TransparentVAE encoder

### Q5: Alpha通道效果不好？
**解决方案：**
```bash
--alpha_loss_weight 2.0  # 增大权重
--loss_type l1  # 换用L1 loss
```

## 总结

**Pixel-Space Loss训练的核心思想：**
- 直接优化最终图像质量
- 更直观、更可控
- 适合精细调整和小数据集

**权衡：**
- 训练速度换取更好的优化目标
- 显存开销换取更精确的控制
- 适合追求质量而非效率的场景

根据你的需求选择合适的训练模式：
- **快速迭代 / 大数据集** → 使用 `train_style_lora.py` (latent loss)
- **精细调整 / 高质量** → 使用 `train_style_lora_pixel_loss.py` (pixel loss)
