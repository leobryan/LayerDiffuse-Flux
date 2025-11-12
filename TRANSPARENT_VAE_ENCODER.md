# TransparentVAE Encoder 说明

## 概述

TransparentVAE使用双重编码策略来处理透明图像：

1. **标准VAE** - 编码RGB内容
2. **透明度Offset Encoder** - 编码Alpha通道和透明区域填充信息

## 工作原理

### TransparentVAE.encode() 流程

```python
def encode(self, img_rgba, img_rgb, padded_img_rgb, use_offset=True):
    # Step 1: 使用标准Flux VAE编码RGB内容
    latent_dist = self.sd_vae.encode(img_rgb).latent_dist
    # 得到: mean 和 std

    # Step 2: 使用透明度offset encoder处理透明度信息
    offset_feed = concat([padded_img_rgb, alpha_channel])  # 4通道输入
    offset = self.encoder(offset_feed) * self.alpha  # alpha=300.0

    # Step 3: 融合透明度信息到latent
    if use_offset:
        latent = mean + std * offset  # ← 关键！包含透明度信息
    else:
        latent = mean + std * random_noise  # 忽略透明度

    return latent
```

### 关键点

**如果 `use_offset=True` (默认):**
- ✅ 使用TransparentVAE的encoder编码透明度
- ✅ Latent包含RGB内容 + 透明度信息
- ✅ 训练出的模型可以生成透明图像

**如果 `use_offset=False`:**
- ❌ **忽略透明度encoder**
- ❌ Latent只包含RGB内容（类似标准VAE）
- ❌ 模型无法学习透明度特征

## 训练参数

### 默认行为（推荐）

```bash
# TransparentVAE encoder 默认启用
python train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data"
    # use_offset=True (默认)
```

### 禁用透明度encoder（不推荐）

```bash
# 仅用于调试或对比实验
python train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --disable_offset  # ← 禁用透明度编码
```

## 输入数据

TransparentVAE.encode() 需要三个输入：

| 输入 | 格式 | 范围 | 说明 |
|------|------|------|------|
| `img_rgba` | (B, 4, H, W) | [0, 1] | 完整RGBA图像 |
| `img_rgb` | (B, 3, H, W) | [-1, 1] | Premultiplied RGB (rgb * alpha, 归一化) |
| `padded_img_rgb` | (B, 3, H, W) | [0, 1] | 透明区域填充的RGB (通过alpha pyramid) |

### 数据预处理（dataset.py中的处理）

```python
# 1. 加载RGBA图像
img_np = np.array(image)  # (H, W, 4) uint8
img_np_01 = img_np / 255.0  # [0, 1]

# 2. 生成padded_rgb - 使用alpha pyramid填充透明区域
padded_rgb = pad_rgb(img_np_01)  # 输入[0,1]，输出[0,1]

# 3. 生成img_rgba
img_rgba = torch.from_numpy(img_np_01).permute(2, 0, 1)  # [0, 1]

# 4. 生成img_rgb - premultiply alpha并归一化
rgb = img_rgba[:3]
alpha = img_rgba[3:4]
img_rgb = rgb * alpha  # Premultiply
img_rgb = img_rgb * 2.0 - 1.0  # [-1, 1]
```

## Transparency Offset Encoder 架构

```python
class LatentTransparencyOffsetEncoder(torch.nn.Module):
    def __init__(self, latent_c=4):
        self.blocks = nn.Sequential(
            Conv2d(4, 32),   # 输入: [padded_rgb(3ch), alpha(1ch)]
            SiLU(),
            Conv2d(32, 32),
            SiLU(),
            Conv2d(32, 64, stride=2),   # 8x downsampling
            SiLU(),
            Conv2d(64, 64),
            SiLU(),
            Conv2d(64, 128, stride=2),
            SiLU(),
            Conv2d(128, 128),
            SiLU(),
            Conv2d(128, 256, stride=2),
            SiLU(),
            Conv2d(256, 256),
            SiLU(),
            zero_module(Conv2d(256, latent_c)),  # 输出offset
        )
```

**特点：**
- 8x下采样 (Flux VAE是16x，所以输入需要128的倍数)
- Zero-initialized最后一层（训练初期offset接近0）
- 输出通道数匹配latent维度（16 for Flux）

## Alpha参数

`alpha = 300.0` - offset的缩放系数

```python
offset = self.encoder(offset_feed) * self.alpha
```

**作用：**
- 类似LoRA的alpha，避免初始zero-initialized输出过小
- 调整透明度信息的强度
- 默认300.0来自原论文/实现

**如果遇到NaN：**
- Alpha值过大可能导致数值溢出
- 可以尝试减小（如30.0, 10.0）进行调试

## 日志输出

训练时会显示：

```
Loading TransparentVAE from ./models/TransparentVAE.pth
Transparency offset encoder: ENABLED
  - TransparentVAE encoder will be used during latent encoding
  - Alpha value for offset: 300.0
```

如果看到 `DISABLED`，说明透明度encoder被禁用（不推荐）。

## 常见问题

### Q1: 训练的模型无法生成透明图像
**可能原因：** 训练时没有使用transparency encoder

**检查：**
1. 训练日志中是否显示 `Transparency offset encoder: ENABLED`
2. 训练命令中是否错误地加了 `--disable_offset`

### Q2: 训练时出现NaN loss
**可能原因：** Transparency encoder产生的offset过大

**调试步骤：**
1. 运行 `python test_channel_ordering.py` 检查encoder输出
2. 运行 `python debug_vae_encoding.py` 检查每一步的数值
3. 如果offset过大，考虑修改alpha值或检查输入数据

### Q3: 如何验证TransparentVAE正常工作
**方法：**
```bash
# 运行原repo的demo，确认TransparentVAE.pth正确
python demo_t2i.py \
    --ckpt_path "./models/flux-dev" \
    --lora_weights "./models/layerlora.safetensors" \
    --trans_vae "./models/TransparentVAE.pth" \
    --prompt "glass bottle"
```

如果demo能生成透明图像，说明TransparentVAE.pth是正确的。

## 总结

✅ **默认配置（推荐）：**
- TransparentVAE encoder **启用**
- 可以学习透明度特征
- 生成的模型支持透明图像

❌ **禁用encoder（不推荐）：**
- 仅用于调试或对比实验
- 模型无法学习透明度
- 等同于标准Flux fine-tuning
