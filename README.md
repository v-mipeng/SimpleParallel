# Readme for SimpleParallel

# 简介

目前业界有很多开源框架实现并行计算，比如PaddlePaddle，Megatron-LM，DeepSpeed，Accelerator。但这些框架一般封装较深，同时为了应对各种可能的情况会引入很多分支代码，导致代码内容繁冗复杂。如果我们想通过阅读这些框架的源代码来熟悉其背后的并行计算原理，负担较重。

本工作是我们基于开源代码进行提炼、修改得到一个能够用于单机多卡张量并行计算的实现。本工作的实现原则是用尽可能少的代码，尽可能少地依赖封装库，在展示并行计算细节的前提下，实现一个可正确运行的并行计算系统。

本工作的适用对象是那些想尽可能多地了解并行计算机制的初学者或者喜欢DIY和深度操控模型训练过程的研究人员。本工作的缺点在于没有实现很多并行计算优化方法 (比如pipeline parallism 的异步计算)，如果大家有兴趣，可以逐步引入其他的并行机制，进行深度DIY。

# 快速开始

## 结构说明

```text
SimpleParallel/
├── README.md # 文档说明
├── train_tokenizer.py # 基于sentencepiece训练一个中文分词模型
├── pre_tokenize_data.py # 将预训练文本切割成token_id序列并存储到文件中
├── main.py # 系统入口，引入配置参数，整合数据、模型、优化器、训练过程接口，启动模型训练过程
├── model.py # 基于张量并行的LLaMA模型实现（参考自LLaMA开源代码）
├── data.py # 加载基于pre_tokenize_data.py得到的id序列文件并生进行训练、验证、和测试数据
├── config.yaml # 配置文件，将被main.py调用
├── utils/ 
	├── initialize.py # 生成并行组的全局参数，方便调用
	├── mapping.py # 定义all_reduce等操作的forward-backward过程（主要是定义backward过程，从而实现多卡梯度并行计算）
	├── utils.py # 定义了print_rank_0，grad_norm函数
	└── layer.py # 定义了一些基于张量并行的基础模块，如VocabularyEmbedding，Linear
```

## 代码使用

1. Option: 首先使用train_tokenizer.py 训练一个sentencepiece分词模型
2. Option: 用pre_tokenize_data.py预处理文本数据，进行id化
3. Option: 配置config文件 (当前配置能在resource提供的测试数据上运行)
4. cd到run.sh同目录，运行 sh run.sh

## 必看代码

1. utils/layers.py
2. main.py
3. model.py
