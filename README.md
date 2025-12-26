# sglang_min_piecewise (Minimal)

目标：提供一个 **SGLang 风格的最小实现**，用于做“混合 Piecewise CUDA Graph”优化：

- 使用 **torch.fx** 自动将一个大的 `forward` 按算子/模块类型切成多个 **piece (子图)**
- 对“支持 CUDA Graph 的 piece”（如 Embedding / MLP / 简单算子链）进行 **CUDA Graph capture + replay**
- 对“不适合/不想 capture 的 piece”（典型是 Attention）保留 **eager**
- 核心是在 **兼容性** 与 **性能** 之间取得平衡

> Code comments are in English by design; docs can be Chinese.

## 核心 API

- `PiecewiseHybridConfig`: 配置 capture sizes、warmup、如何识别 attention。
- `make_piecewise_hybrid_model(model, config, example_inputs_fn, ...)`:
  - 将 `model` FX trace + split + 子模块替换
  - 返回一个可运行的 `nn.Module`（内部有 capture/replay/eager 混合）

## 重要约束（刻意简化）

1. 只支持 **单一动态维度** 的分桶（默认按第一个 tensor 入参的 `shape[0]`）。
2. 仅实现最常见的“token-major”布局：`[T, ...]`。
3. 为了让“Attention piece 保持 eager”真正生效：trace 阶段会把 `is_attention_module(...)==True` 的模块
  当作 **leaf module**，确保 FX 图里存在一个 `call_module(attn_*)` 节点，从而可以被 splitter 隔离成单独 piece。
  否则（例如自定义 Attention Module 被 inline 展开成 matmul/softmax），attention 可能被错误地 capture，
  在 `static_size > runtime_size` 时会因为 token mixing 导致数值不一致。
4. 若你的真实模型是 `[B, L, ...]`（动态在 dim=1），你需要：
   - 在业务侧改成 `[T, ...]` 传入被 capture 的片段，或
   - 自定义 `runtime_size_fn`（以及配套的 copy/slice 策略）。
5. 这是“最小可跑”的教学/原型实现，不覆盖 SGLang 的全部工程化细节（fake tensor、SymInt、分布式通信、复杂缓存结构等）。

6. **仅面向推理（inference-only）**：在完成 `capture()` 并安装 CUDAGraph backends 后，
  `PiecewiseHybridModel.forward` 会在 `torch.inference_mode()` 下执行整条 stitched graph，
  避免 inference tensor 与 grad-enabled eager op 混用导致的报错。

补充：在 capture 时，框架会先用 `torch.empty/empty_like` 分配静态 buffer（保证地址稳定），
然后把录制到的运行时输入 `copy_` 进静态 buffer 再做 warmup/capture。
这一步是必须的，否则像 `nn.Embedding` 这类算子可能读到未初始化的随机 index，触发 CUDA device-side assert。

## 快速测试

在该目录运行：

```bash
pytest -q
```

如果没有 CUDA，会自动 skip CUDA tests。
