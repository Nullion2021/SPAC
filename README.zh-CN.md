# SPAC：面向分布式神经推理的会话绑定路径认证

**[English](README.md)** | **[中文](README.zh-CN.md)**

**论文（即将上线）** | **[中文项目主页](https://nullion2021.github.io/SPAC/index_zh.html)** | **[English Page](https://nullion2021.github.io/SPAC/)**

SPAC（Session-bound Path Authentication Chain，会话绑定路径认证链）是一种面向分布式神经网络推理的链式群体认证机制。它将设备身份、有序逐跳记录、已注册的分片声明与跨跳编码激活连续性绑定到同一份可验证证明中，把"指定的分布式推理确实在我的设备上执行过"转化为一条可校验的认证声明，可抵御信任边界上的代理模型替换与拓扑破坏攻击。

> 分布式神经网络推理在彼此独立管理的设备上依次执行有序的模型分片，因此需要对会话参与和激活传递进行认证。在至多 k 台（k < N）设备可能被腐化的博弈模型下，通过归约到消息认证码（MAC）不可伪造性，可以界定诚实设备参与被全新误归因的上界。在所测试的代理攻击、拓扑攻击与摘要不一致注入负载中，均未观察到被接受的伪造；SPAC 在所评估的 MLP 与 ResNet 配置下仅增加 0.65–2.56 毫秒开销。

## 实验结果概要

| 实验维度 | 结果 |
|---------|------|
| 代理模型攻击 | 252 个测量单元中 SPAC FAR = 0；仅输出比对的基线对完整模型替代的接受率 ASR = 1 |
| 拓扑破坏攻击 | 4 种攻击 × 3 种链深度，FAR = 0 |
| 激活注入 | 所覆盖的全部 MLP / ResNet-18 / ViT 配置中 SPAC 接受事件数为零（每单元单侧 95% 上界 0.93%） |
| 量化稳定性 | 共享数值配置下 QMatch = 1.0，FRR = 0 |
| 系统开销 | 每会话 +0.65–2.56 毫秒；证明元数据随链长线性增长 |

## 仓库结构

```text
├── docs/                    # 项目主页（GitHub Pages 静态站点，中英双语）
├── src/group_auth_dnn/      # 可复用软件包
│   ├── protocol/            # 核心协议：编码、度量、会话、链、节点
│   ├── models/              # 切分模型定义：MLP、ResNet-18、ViT
│   └── ...
├── experiments/             # 保留的实验驱动（主实验 + 消融）
├── scripts/                 # 复现入口脚本
├── tests/                   # 发布检查
├── data/                    # 本地数据集（由准备脚本创建）
├── checkpoints/             # 本地预训练权重（由准备脚本创建）
└── results/                 # 生成的 CSV / JSON / 图表
```

## 快速开始

推荐 Python 3.10 及以上版本。

```bash
git clone https://github.com/Nullion2021/SPAC.git
cd SPAC
pip install -r requirements.txt
```

准备本地数据集与可选的预训练权重：

```bash
python scripts/prepare_data.py
python scripts/prepare_checkpoints.py
```

运行主实验（代理攻击、拓扑破坏、量化稳定性、系统开销）：

```bash
python scripts/run_main_experiments.py
```

运行关键消融实验（度量绑定、新鲜性、激活链、链状态、签名模式）：

```bash
python scripts/run_ablations.py
```

生成论文图表与发布摘要表：

```bash
python scripts/generate_figures.py
```

实验产物输出至 `results/`。各入口脚本的详细说明见
[`scripts/README.md`](scripts/README.md)。

## 输入要求

- CIFAR-10 数据集放在 `data/cifar10/`（部分驱动可自动下载）
- 可选预训练权重放在 `checkpoints/mlp/`、`checkpoints/resnet/`、`checkpoints/transformer/`、`checkpoints/vit/`

## 项目主页

学术项目主页由本仓库的 [`docs/`](docs/) 目录托管：
- 中文：<https://nullion2021.github.io/SPAC/index_zh.html>
- English: <https://nullion2021.github.io/SPAC/>

页面基于 [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) 构建。

## 引用

如果你使用了本代码，请引用：

```bibtex
@misc{xiang2026spac,
  title        = {SPAC: Session-Bound Path Authentication for Distributed Neural Inference},
  author       = {Xiang, Zhengzhe and Chen, Yinlin and Cheng, Guanjie and Taheri, Javid and Deng, Shuiguang},
  year         = {2026},
  howpublished = {Manuscript submitted to IEEE Transactions on Dependable and Secure Computing},
  url          = {https://nullion2021.github.io/SPAC/}
}
```

## 许可证

代码以 [Apache License 2.0](LICENSE) 发布（另见 [NOTICE](NOTICE)）。
