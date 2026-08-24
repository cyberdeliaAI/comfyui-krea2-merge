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
- 支持从 `-4.0` 到 `4.0` 的正负权重。
- 默认保存到 `ComfyUI/models/loras/krea2-merged-loras`。

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

## 兼容性

`exact_concat` 是默认模式，可精确合并完整的 A/B 或 down/up 对，并避免分别
相加 LoRA 因子产生的交叉项。它也支持不同 rank；例如 rank 4 与 rank 16 会
输出 rank 20。`legacy_linear` 仅用于兼容旧工作流，它是近似算法，并要求共享
键具有相同的形状和 rank。`exact_concat` 直接使用权重，并忽略
`force_same_strength`。两种模式都不会修改 Krea 2 底模。当 PEFT 文件不包含
alpha 时，本扩展使用 `alpha = rank`。DoRA、LoCon `lora_mid` 和其他不支持的
adapter 会明确报错，而不会被静默丢弃或错误合并。

## 致谢

本项目基于
[LingSss9/comfyui-merge](https://github.com/LingSss9/comfyui-merge)，并保留原 MIT 许可证。
