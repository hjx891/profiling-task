# 02 · 走进一条完整的图像训练流水线

这一回，我们换一个真正更复杂的任务：从合成图像取出 batch，做预处理，训练带残差块的 CNN，再执行一次评估。它与第一部分的 MLP 推理是不同任务。

| 文件 | 职责 |
| --- | --- |
| `main.py` | 初始化、预热、组织采集、保存结果 |
| `config.py` | 参数与合法性检查 |
| `data.py` | 合成带类别特征的图像、DataLoader、增强、归一化、设备传输 |
| `model.py` | stem、两个残差块、降采样、全局池化与分类头 |
| `engine.py` | 训练步骤、交叉熵、反向、Adam 更新、评估 |

在仓库根目录运行：

```bash
python 02-pytorch-multi/main.py
python 02-pytorch-multi/main.py --image-size 48 --output results/pytorch-multi-image48
```

默认 batch=16、图像32×32、预热2个训练步骤、采集8个训练步骤和1次评估、CPU线程数2。初始化、合成完整数据集和预热在采集外；DataLoader 取 batch 与预处理在采集内。没有网络、磁盘数据集和多进程 worker。

打开 `results/pytorch-multi/trace.json`，尝试展开：

```text
training_pipeline
├── load_batch
├── train_step
│   ├── preprocess
│   │   ├── augmentation
│   │   └── normalize
│   ├── device_transfer
│   ├── zero_grad
│   ├── forward
│   │   ├── stem
│   │   ├── backbone
│   │   │   ├── residual_block_1 → conv_path / residual_add → 算子
│   │   │   └── residual_block_2 → conv_path / residual_add → 算子
│   │   └── classification_head
│   ├── loss
│   ├── backward
│   └── optimizer_step
└── evaluation
    ├── preprocess / device_transfer
    ├── eval_forward → 模型各模块
    └── metrics
```

这些区间由真实执行的 `record_function` 标记，不是按文件名伪造的调用关系。CPU 上的标记不等于 GPU kernel 的准确边界；CUDA 同学还要对照设备轨道。注意父阶段的 total 含子阶段，不能相加。

请回答：前向、反向和更新各自做了什么？同样的模型为何在训练和评估中出现两次？图像尺寸改变时，哪些计算量会变化？找一条至少四层的真实嵌套路径，分别定位到这些文件。

数据量与训练步数很小，`eval_accuracy` 是流程结果，不是模型在真实数据集上的性能。我们要练习看清流程，不是在这个任务里刷准确率。


问题 1：前向、反向和更新各自做了什么？
答：从数据可以看出，forward（前向）包含 convolution 等算子（约占 21%）；backward（反向）包含 convolution_backward 等算子（约占 32%）；optimizer_step（参数更新）包含 Adam.step。这说明反向传播是训练阶段最耗时的环节，前向次之，参数更新本身耗时较少（只占 10% 左右）。

问题 2：同样的模型为何在训练和评估中出现两次？
答：排行榜里出现了 argmax（用来算准确率）和 mean（算平均损失），这说明评估阶段确实被执行了。由于评估阶段只做前向传播（没有 backward），所以我们在图里能明显看到 evaluation 下的调用链要短得多，没有反向求导的算子。

问题 3：图像尺寸改变时，哪些计算量会变化？
答：如果改变图像尺寸，最上方的 convolution（卷积）和 convolution_backward（卷积反向传播）的耗时会发生显著变化。因为图像变大，卷积核滑动的次数和计算量都会成倍增加。调用次数虽然不变，但单次算子时间会变长。

问题 4：找一条至少四层的真实嵌套路径，分别定位到这些文件
答：我在图中找到了这条路径：
training_pipeline → train_step → forward → backbone → residual_block_1 → conv_path。
通过对照源码：前两层在 engine.py 和 main.py 里组织流程；后几层在 model.py 里定义模型结构。在 conv_path 这个框里，我能看到底层的 mkldnn_convolution 算子正在执行。
