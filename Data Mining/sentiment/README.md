### 实验环境

```
centos 7.4
python 3.6.7
configparser 3.7.4
numpy 1.15.4
tqdm 4.32.1
scikit-learn 0.20.2
torch 1.1.0
torchvision 0.3.0
tensorboradX 1.7
pytorch-pretrained-bert 0.6.2
pyltp 0.2.1
cuda 9.0.176
cudnn 7.3.0
gpu型号: 双核11G的Tesla K80(4块)
```

### 源代码文件介绍

* 预处理

  preprocess.py: 对训练数据预处理,分割出验证机并随即打乱顺序

* baseline

  baseline.py: 用pyltp分词,然后使用朴素贝叶斯进行分类

* 主程序

  * sentiment_beta.py：不按各类型数据比例调整损失权重，不加验证集筛选存储的模型
  * sentiment.py：按各类型数据比例调整损失权重，不加验证集筛选存储的模型
  * sentiment_dev.py：按各类型数据比例调整损失权重，加验证集筛选存储的模型

* 运行脚本

  run_sentiment.sh: 具体参数在下面会说明

### 实验运行

* Bert预测

实验参数配置写在src目录下的run_sentiment.sh脚本中，内容如下

```bash
#! /usr/bin/env bash
python sentiment.py \
  --data_dir '../input' \
  --bert_model_dir '../input/pre_training_models/chinese_L-12_H-768_A-12' \
  --output_dir '../output/models/' \
  --max_seq_length 300 \
  --do_predict \
  --do_lower_case \
  --train_batch_size 60 \
  --gradient_accumulation_steps 3 \
  --predict_batch_size 15 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --checkpoint '../output/models/checkpoint-33426'
```

上面为运行预测模型的脚本内容，命令行下进入src目录执行如下命令即可开始预测

```
bash run_sentiment.sh
```

预测结果文件保存在脚本配置项output_dir对应目录下: test.predict-\*\*\*\*\*

* Bert训练

训练时需要将run_sentiment.sh文件中的checkpoint配置项注释掉,将do_predict改为do_train，改完之后对应内容如下

```bash
#! /usr/bin/env bash
python sentiment.py \
  --data_dir '../input' \
  --bert_model_dir '../input/pre_training_models/chinese_L-12_H-768_A-12' \
  --output_dir '../output/models/' \
  --max_seq_length 300 \
  --do_train \
  --do_lower_case \
  --train_batch_size 60 \
  --gradient_accumulation_steps 3 \
  --predict_batch_size 15 \
  --learning_rate 2e-5 \
  --num_train_epochs 3
  #--checkpoint '../output/models/checkpoint-33426' \
```

然后命令行进入到src目录下执行如下命令即可从零开始训练，如果想从某个checkpoint开始继续训练，则需在脚本中指定checkpoint配置项

```bash
bash run_sentiment.sh
```

训练过程中会将在验证集上F1 score最高的5个模型保存在脚本配置项output_dir对应目录下: checkpoint-\*\*\*\*\*

训练和预测过程可以在cpu也可以在gpu上跑，有gpu的情况下程序会自动识别gpu并将数据并行划分到device_id为0和1的gpu上，因此要确保有gpu的机器上至少有双核才能保证程序能正常运行。

* baseline

baseline使用到了pyltp用于分词和词性标注，因此要运行还需要下载ltp_model_v3.4.0模型到input目录下(有点大我就没上传了)，然后命令行进入到src目录下执行如下命令

```bash
python baseline.py
```

预测结果写在output目录下的baseline.csv中

