# 说明文档

## 一、项目结构

```
ner
├── input
│   └── renmin1998
│       ├── tag2id.bmoe               序列标签
│       ├── test.char.bmoe          测试集
│       ├── train.char.bmoe        训练集
│       ├── val.char.bmoe            验证集  
│       └── vocab_char.txt           词汇表
├── output
│   └── renmin1998_bmoe                 
│       └── ckpt                                            模型保存目录
│           └── bilstm_crf_0.pth.tar        可加载的模型
└── src                                     源码目录
    ├── data_loader.py        数据加载模块
    ├── decoder.py                解码器模块
    ├── encoder.py                编码器模块
    ├── main.py                      主文件运行入口
    ├── metrics.py                 评估模块
    ├── model.py                   模型模块
    ├── run.sh                          运行脚本
    ├── trainer.py                   训练框架模块
    └── utils.py                        工具库模块
```

## 二、运行步骤

运行脚本run.sh内容如下：

```
CUDA_VISIBLE_DEVICES=0 \
python main.py \
    --dataset renmin1998 \
    --compress_seq \
    --tagscheme bmoe \
    --use_lstm \
    --use_crf \
    --embedding_size 300 \
    --hidden_size 300 \
    --batch_size 64 \
    --dropout_rate 0.1 \
    --lr 1e-3 \
    --max_length 100 \
    --max_epoch 20 \
    --warmup_epoches 0 \
    --optimizer adam \
    --metric micro_f1 \
    # --only_test \
```

在命令行下执行bash run.sh即可训练模型，模型训练过程中会保存在验证集上结果最好的模型到ckpt目录下。在脚本中去掉--only_test的注释即可加载最近保存的模型在测试集上进行测试。通过配置run.sh中的命令行参数即可实现对模型的调参，各项参数的含义在main.py中有说明。