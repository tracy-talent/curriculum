# 基于BiLSTM+CRF实现NER(tensorflow)

## How to launch

> 运行环境：linux
>
> python环境：python3, tensorflow>=1.9



* 进入data目录下：

```shell
# 下载glove词向量包
make download-glove
# 生成python可用的glove
make build-glove
```

* 进入src目录下，执行main.py

```python
python main.py
```

* 最后生成的预测结果存储在src/result/test.preds.txt中