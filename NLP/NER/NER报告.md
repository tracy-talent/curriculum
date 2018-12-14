<center><h1>NER实验报告</h1></center>

<center>日期：2018/12/4</center>



## 一、任务描述

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NER（Named Entity Recognition，命名实体识别）又称作专名识别，是自然语言处理中常见的一项任务，使用的范围非常广。命名实体通常指的是文本中具有特别意义或者指代性非常强的实体，通常包括人名、地名、机构名、时间、专有名词等。NER系统就是从非结构化的文本中抽取出上述实体，并且可以按照业务需求识别出更多类别的实体，比如产品名称、型号、价格等。因此实体这个概念可以很广，只要是业务需要的特殊文本片段都可以称为实体。在本次作业中，我们的任务就是从已经分好词的中文文本中识别出人名（PERSON）、地点（LOCATION）、时间（TIME）及机构名（ORGANIZATION）。



## 二、设计思路与程序分析

### 2.1 设计思路

* 做了简单的调研，发现目前实现命名实体识别采用的最广泛效果最好的方法就是BiLSTM+CRF，因此我决定通过学习这种方法来完成此次的任务。整个设计划分成3个部分：
  1. Word Embeddings: 将中文词语转换成计算机可处理的向量形式，使用stanford开源的已训练好的glove模型对分好的汉词做embedding。
  2. BiLSTM: 先说一下LSTM，LSTM模型神经元信息只能从前向后传递，也就意味着，当前时刻的输入信息仅能利用之前时刻的信息。然而对于序列标注任务来说，当前状态之前的状态和之后的状态应该是平权的。命名实体的标签之间具有强烈的依赖关系，BiLSTM则既能利用当前时刻之前的信息，又能利用之后的信息，通过利用词的上下文信息得到词的向量表示。
  3. CRF：CRF用于对LSTM学到词表示结果进行解码，计算标签得分，使用每个词对应的隐藏状态向量来做最后预测。



### 2.2 程序实现与分析

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;google开源的tensorflow框架提供了lstm和crf的API，而且使用起来要比theano方便简洁得多，因此程序的实现使用了tensorflow。

* 读入词和实体标签文件内容并利用tf.data打包成tensorflow的estimator可以处理的数据集，除此之外还需要建立一个在数据文件中出现过的所有字的数据集，因为待预测的数据可能包含了训练数据集中不存在的词汇，就需要依靠这些字来拼凑成这个词汇：

```python
# 将文件内容并编码成tf可使用的bytes
def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"

    # Chars
    chars = [[c.encode() for c in w] for w in line_words.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
    return ((words, len(words)), (chars, lengths)), tags

# 读入word和tag文件，将行内容传递给parse_fn处理
def generator_fn(words, tags):
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)

# 使用tf.data.Dataset.from_generator接口将generator_fn生成的数据打包，并使用padded_batch
# 对数据进行规整
def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = ((([None], ()),               # (words, nwords)
               ([None, None], [None])),    # (chars, nchars)
              [None])                      # tags
    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)
    defaults = ((('<pad>', 0),
                 ('<pad>', 0)),
                'O')
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset
```

然后在main函数中调用input_fn生成训练集和验证集：

```python
train_inpf = functools.partial(input_fn, fwords('train'), ftags('train'),
                                   params, shuffle_and_repeat=True)
eval_inpf = functools.partial(input_fn, fwords('testa'), ftags('testa'))
```

* 有了数据集以后就可以需要实例化一个estimator对生成的数据集进行训练和验证以及预测，在实例化之前还需要做一些准备工作，建立一个超参字典，既是模型需要也方便后期我们手动调参：

```python
params = {
        'dim': 300,   # 词向量维度
        'dropout': 0.5,  # dropout通常设0.5最佳
        'num_oov_buckets': 1,
        'epochs': 25,   # 训练轮数
        'batch_size': 20,  
        'buffer': 15000,  # shuffle buffer
    	'char_lstm_size': 25, # 字的lstm隐藏层数
        'lstm_size': 100,  # 词的lstm隐藏层数
        'words': str(Path(DATADIR, 'vocab.words.txt')),  # 词文件，每行一个
        'chars': str(Path(DATADIR, 'vocab.chars.txt')),  # 字文件，每行一个
        'tags': str(Path(DATADIR, 'vocab.tags.txt')),  # 词实体标签文件，每行一个
        'glove': str(Path(DATADIR, 'glove.npz')) # 提前训练好的词向量模型glove
    }
```

* 然后estimator的参数配置：

```python
# 每120s保存一次模型
cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
```

* 最后也是最重要的esitimator要跑的模型的设计，写一个模型函数def model_fn(features, labels, mode, params)，在里面组建自己想要的模型，其中features是包含要处理的词和字的数据文件，labels是数据对应的标签，mode包含train, evaluate, predict三种模式，params就是之前设置的超参，下面说一下model_fn的实现，首先从features读出数据：

```python
	# For serving features are a bit different
    if isinstance(features, dict):
        features = ((features['words'], features['nwords']),
                    (features['chars'], features['nchars']))

    # Read vocabs and inputs
    (words, nwords), (chars, nchars) = features
    dropout = params['dropout']
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    vocab_chars = tf.contrib.lookup.index_table_from_file(
        params['chars'], num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1
    with Path(params['chars']).open() as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']
```

* 然后使用预训练好的glove作chars embeddng:

```python
# Char Embeddings
char_ids = vocab_chars.lookup(chars)
variable = tf.get_variable(
    'chars_embeddings', [num_chars, params['dim_chars']], tf.float32)
char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout,
                                    training=training)
```

* 然后构建Chars LSTM模型：

```python
# Char LSTM
dim_words = tf.shape(char_embeddings)[1]
dim_chars = tf.shape(char_embeddings)[2]
flat = tf.reshape(char_embeddings, [-1, dim_chars, params['dim_chars']])
t = tf.transpose(flat, perm=[1, 0, 2])
lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['char_lstm_size'])
lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['char_lstm_size'])
lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
_, (_, output_fw) = lstm_cell_fw(t, dtype=tf.float32,
                                 sequence_length=tf.reshape(nchars, [-1]))
_, (_, output_bw) = lstm_cell_bw(t, dtype=tf.float32,
                                 sequence_length=tf.reshape(nchars, [-1]))
output = tf.concat([output_fw, output_bw], axis=-1)
char_embeddings = tf.reshape(output, [-1, dim_words, 50])
```

* 然后使用预训练好的glove作word embedding，并concat Chars embedding和Words embedding:

```python
# Word Embeddings
word_ids = vocab_words.lookup(words)
glove = np.load(params['glove'])['embeddings']  # np.array
variable = np.vstack([glove, [[0.] * params['dim']]])
variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

# Concatenate Word and Char Embeddings
embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)
```

* 然后调用LSTM模型对embedding进行学习：

```python
# LSTM
t = tf.transpose(embeddings, perm=[1, 0, 2])  # Need time-major
lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
output = tf.concat([output_fw, output_bw], axis=-1)
output = tf.transpose(output, perm=[1, 0, 2])
output = tf.layers.dropout(output, rate=dropout, training=training)
```

* 最后一步调用CRF对LSTM学到的次序列标签打分完成解码任务：

```python
# CRF
logits = tf.layers.dense(output, num_tags)
crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)
```

* 最后依据不同的mode作不同的处理：

```python
if mode == tf.estimator.ModeKeys.PREDICT:
    # Predictions
    reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
        params['tags'])
    pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
    predictions = {
        'pred_ids': pred_ids,
        'tags': pred_strings
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
else:
    # Loss
    vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
    tags = vocab_tags.lookup(labels)
    log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
        logits, tags, nwords, crf_params)
    loss = tf.reduce_mean(-log_likelihood)

    # Metrics
    weights = tf.sequence_mask(nwords)
    metrics = {
        'acc': tf.metrics.accuracy(tags, pred_ids, weights),
        'precision': precision(tags, pred_ids, num_tags, indices, weights),
        'recall': recall(tags, pred_ids, num_tags, indices, weights),
        'f1': f1(tags, pred_ids, num_tags, indices, weights),
    }
    for metric_name, op in metrics.items():
        tf.summary.scalar(metric_name, op[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    elif mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer().minimize(
            loss, global_step=tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)
```

* 完成最难的model_fn的设计之后，即可启动estimator进行训练和评估了：

```python
# 为防止过拟合设置hook让训练early stopping
hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```



## 三、实验总结

* 通过本次实验让我对深度学习在NLP的应用有了一个初步的了解，没想到BiLSTM+CRF能够让命名实体识别的F1度量达到95%以上，可见深度学习应用到NLP上还是效果不错的。目前NLP领域研究最热门的方法就是深度学习，后续的学习过程中我会进一步去了解、学习更多的深度学习在NLP方面的应用。
* 由于之前没有学习过深度学习和tensorflow，所以本次实验引用和参考了：[https://github.com/guillaumegenthial/tf_metrics](https://github.com/guillaumegenthial/tf_metrics)，在此对作者表示感谢