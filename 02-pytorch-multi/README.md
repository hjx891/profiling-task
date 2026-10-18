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
forward（前向传播）：把输入数据（图像）喂给模型，一层一层计算，得到预测结果。这是“做菜”的过程。

backward（反向传播）：把预测结果和真实标签的差距（损失），从输出层往回传，计算每个参数“应该往哪个方向调整”。这是“复盘哪里做得不好”。

optimizer_step（参数更新）：根据反向传播算出的梯度，真正地去修改模型的参数（权重）。这是“纠正错误、改进做法”。

这三个是并列的训练阶段，不是谁包含谁。

问题 2：同样的模型为何在训练和评估中出现两次？
因为这是两个不同的阶段：

第一次出现在 train_step → forward，是为了训练——前向计算后还要接 backward 和 optimizer_step，目的是学习。

第二次出现在 evaluation → eval_forward，是为了评估——只做前向计算，不更新参数，目的是检查模型当前的水平。

evaluation 里的 eval_forward 虽然和训练时的 forward 是同一个模型结构，但它在调用树上位于 evaluation 分支下，而不是 train_step 分支下。

问题 3：图像尺寸改变时，哪些计算量会变化？
图像尺寸从 32×32 变成 48×48，图像的总像素数变多了。这会影响到：

load_batch 和 preprocess 中处理的数据量变大；

forward 里卷积层（conv_path）的计算量会显著增加，因为卷积核要在更大的特征图上滑动；

相应地，backward 的计算量也会增加。

调用次数不会变（模型结构没变），但每次算子执行的时间会变长。这和单文件实验里 batch 变大的逻辑是一致的。

问题 4：找一条至少四层的真实嵌套路径，分别定位到这些文件
例如这条路径：

text
training_pipeline → train_step → forward → backbone → residual_block_1 → conv_path
training_pipeline / train_step：在 main.py（组织流程）和 engine.py（训练步骤逻辑）中；

forward / backbone / residual_block_1 / conv_path：在 model.py 中定义（模型结构）。

再比如：

text
training_pipeline → evaluation → eval_forward → backbone → residual_block_1
前半段（evaluation、eval_forward）在 engine.py；

后半段（backbone、residual_block_1）在 model.py。

每展开一层，就打开对应的源码文件看一眼，确认这个区间在源码里到底负责什么。这是 Profile 分析的核心动作——把图上的方块和代码里的职责对应起来。
