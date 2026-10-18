# 01 · 用一个文件，看清一次 MLP 推理

先从一个容易看明白的例子开始：固定输入，跑一个两层全连接网络。没有数据下载、反向传播和参数更新，所有代码都在 `profile_single.py`。

在仓库根目录、已激活 PyTorch 环境的 Linux/macOS 终端运行：

```bash
python 01-pytorch-single/profile_single.py
python 01-pytorch-single/profile_single.py --batch-size 128 --output results/pytorch-single-batch128
```

默认 batch=64、预热4次、采集12次、CPU线程数2；输入维度256、隐藏维度512、输出维度64。`--device cuda` 仅供已配置好 CUDA 及设备采集支持的 Linux 环境使用。

打开输出目录中的 `trace.json`，找到：

```text
inference_step
└── mlp_forward
    ├── Linear / aten::linear / aten::addmm 等实际算子
    ├── ReLU
    └── Linear
```

实际算子名称与实现可能随 PyTorch 版本、设备和后端变化。`operators.txt` 按 Self CPU 排序；`summary.json` 保存运行参数、环境、输出形状和校验和。

请回答：哪个算子值得关注？它对应哪一层？batch 改变后，调用次数、算子时间和输入规模有什么关系？这里测的是固定输入的前向推理，不能把答案直接套到训练或数据加载上。
问题 1：aten::addmm 最值得关注。它的 Self CPU 占比高达 75.37%，是绝对的性能瓶颈。对应模型里的全连接层（Linear 层）。

问题 2：batch 改变后，调用次数、算子时间和输入规模有什么关系？
我们对比一下你的两组数据：

调用次数：完全没变！

batch=64 时：aten::addmm 调用了 24 次。

batch=128 时：aten::addmm 还是 24 次。

为什么：因为模型的架构没变，还是那两层。不管来多少人，菜谱不变，炒菜的步骤就不会变。

单次算子时间：几乎翻倍！

batch=64 时：单次平均耗时 62.482us。

batch=128 时：单次平均耗时 119.147us。

为什么：因为 batch 翻倍了，矩阵变大了（计算规模变大），算一次的时间自然拉长。你注意看总时间，从 1.403ms 涨到了 2.657ms，非常合理。

输入规模：变大了。

为什么：输入张量的形状从 (64, 256) 变成了 (128, 256)（你可以在 summary.json 里验证这一点）。
