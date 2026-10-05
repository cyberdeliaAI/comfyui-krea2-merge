[English](README.md) | 中文

# ComfyUI Krea2 Merge

ComfyUI Krea2 Merge 是一个独立的 ComfyUI 扩展，无需加载或修改底模即可精确
合并和保存 2 到 4 个 Krea 2 LoRA。它支持 Krea 2 使用的 PEFT/Diffusers
`lora_A` + `lora_B` 格式，同时保留对 Kohya
`lora_down` + `lora_up` 格式的支持。

本分支使用独立的包名、扩展名、节点 ID、分类和输出目录，可以与原版
`comfyui-merge` 同时安装。

## 功能

- 无需加载底模，可合并 2 到 4 个 Krea 2 LoRA。
- 支持 `lora_A/lora_B` 和 `lora_down/lora_up`。
- 在没有 alpha 张量时从 LoRA rank 推断 alpha。
- 正确处理负权重，并对不兼容的张量形状给出明确错误。
- 可通过 `exact_concat` 模式合并不同 rank 的 LoRA。
- 可选的 `svd_truncate` 模式用最优 SVD 截断限制输出 rank，并报告近似误差。
- 支持从 `-4.0` 到 `4.0` 的正负权重。
- 通过 ComfyUI 的安全加载器加载 `.safetensors`、`.sft` 和 PyTorch checkpoint。
- 仅保存到 `ComfyUI/models/loras/krea2-merged-loras` 及其子目录。
- 即使因子张量使用 FP16 或 BF16，alpha 仍以 float32 保存。

## 安装

将本仓库放入：

```text
ComfyUI/custom_nodes/comfyui-krea2-merge
```

然后重启 ComfyUI。

## 节点

所有节点位于 **Krea2 Merge / LoRA** 分类：

- **Krea2 Merge • Load LoRA**
- **Krea2 Merge • Merge LoRAs**
- **Krea2 Merge • Save LoRA**
- **Krea2 Merge • Apply LoRA**

示例工作流包含来自 `ComfyUI-Custom-Scripts` 的 **Show Text** 节点，用于显示保存路径。
Save 节点本身也已注册为输出节点。
Save 节点的 `allow_overwrite` 可控制是否覆盖已存在的输出文件。
在 Windows 上，覆盖前会先备份原文件；保存失败时会自动恢复。

从 1.2.1 起，输出必须使用相对文件名或子目录，例如
`characters/my_merge.safetensors`。绝对路径、`..` 和指向输出目录外部的符号链接
会被拒绝。旧工作流中的绝对输出路径需要改为相对文件名。
支持的输出扩展名为 `.safetensors`、`.sft`、`.pt`、`.pth`、`.ckpt` 和 `.bin`。

## 兼容性

`exact_concat` 是默认模式，可精确合并完整的 A/B 或 down/up 对，并避免分别
相加 LoRA 因子产生的交叉项。它也支持不同 rank；例如 rank 4 与 rank 16 会
输出 rank 20。`svd_truncate` 先按 `exact_concat` 精确合成，再把每个模块限制在
`target_rank`（默认 32，仅此模式使用）以内：保留合并权重增量的最优低秩近似
（最优 SVD 截断），输出 rank 不再随输入数量增长；合成后 rank 已不超过
`target_rank` 的模块保持精确。结果是确定性的，控制台会报告被截断模块的相对误差。
误差即超出 `target_rank` 的那部分增量：对奇异值快速衰减的 LoRA 很小，对 rank
被充分利用的 LoRA 则较大；追求保真度时请让 `target_rank` 接近输入 rank 之和。
仅支持线性层和 up 为 1x1 的模块，其他情况请使用 `exact_concat`。`legacy_linear` 仅用于兼容旧工作流，它是近似算法，并要求共享
键具有相同的形状和 rank。`exact_concat` 和 `svd_truncate` 直接使用权重，并忽略
`force_same_strength`。所有模式都不会修改 Krea 2 底模。当 PEFT 文件不包含
alpha 时，本扩展假设 `alpha = rank`，即单位缩放；原始训练 alpha 无法仅从因子
恢复，非默认缩放需要先单独提供。alpha 始终保存为 float32，以避免 BF16 舍入。
DoRA、LoCon `lora_mid`、LoHa/LoKr、`diff`/`diff_b`、因子偏置和其他不支持的
adapter 键在两种模式下都会明确报错。

输入应使用相同底模和模块命名方式。两个输入没有共享模块名时会发出警告，
但仍保留各自模块；PEFT/Kohya 的不同命名不会自动转换或合并。
工作流中的 LoRA 文件缺失时会保留原文件名，由 ComfyUI 报错，避免静默换成其他文件。

## 致谢

本项目基于
[LingSss9/comfyui-merge](https://github.com/LingSss9/comfyui-merge)，并保留原 MIT 许可证。
