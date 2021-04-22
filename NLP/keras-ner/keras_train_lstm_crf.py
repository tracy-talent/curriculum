import logging
import tensorflow as tf
import tensorflow_addons as tfa
# tf.enable_eager_execution()
import numpy as np
import os

from keras_model_lstm_crf import BILSTM_CRF
from keras_utils import DataLoader_LSTM
from utils import load_vocabulary
from utils import extract_kvpairs_in_bio
from utils import cal_f1_score

# device
tf.config.set_soft_device_placement(True)
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

# hyper parameter
batch_size = 16
training_epoches = 5
use_crf = True

# set logging
log_file_path = "./ckpt/run.log"
if os.path.exists(log_file_path): os.remove(log_file_path)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
fhlr = logging.FileHandler(log_file_path)
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)

# load vocab
logger.info("loading vocab...")
w2i_char, i2w_char = load_vocabulary("./data/vocab_char.txt")
w2i_bio, i2w_bio = load_vocabulary("./data/vocab_bioattr.txt")

# load dataset
logger.info("loading data...")
train_dataset = DataLoader_LSTM(
    "./data/train/input.seq.char",
    "./data/train/output.seq.bioattr",
    w2i_char,
    w2i_bio, 
).get_dataset(batch_size, buffer_size=10000)

valid_dataset = DataLoader_LSTM(
    "./data/test/input.seq.char",
    "./data/test/output.seq.bioattr",
    w2i_char,
    w2i_bio, 
).get_dataset(batch_size, buffer_size=1)

logger.info("building model...")
model = BILSTM_CRF(embedding_dim=300,
                hidden_dim=300,
                vocab_size_char=len(w2i_char),
                vocab_size_bio=len(w2i_bio))
optimizer = tf.keras.optimizers.Adam()

# checkpoint
ckpt_save_path = './ckpt/train'
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_save_path, max_to_keep=3)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    logger.info("Latest checkpoint restored!!")
logger.info("start training...")


def valid(data_processor):
    preds_kvpair = []
    golds_kvpair = []
    
    for inputs_seq_batch, inputs_seq_len_batch, outputs_seq_batch in valid_dataset:
        logits_seq_batch = model(inputs_seq_batch, inputs_seq_len_batch, outputs_seq_batch)
        if use_crf:
            preds_seq_batch, crf_scores = tfa.text.crf_decode(logits_seq_batch, model.transition_params, inputs_seq_len_batch)
        else:
            preds_seq_batch = tf.argmax(logits_seq_batch, axis=-1)
        for pred_seq, gold_seq, input_seq, l in zip(preds_seq_batch, 
                                                    outputs_seq_batch, 
                                                    inputs_seq_batch, 
                                                    inputs_seq_len_batch):
            pred_seq = [i2w_bio[i] for i in pred_seq.numpy()[:l]]
            gold_seq = [i2w_bio[i] for i in gold_seq.numpy()[:l]]
            char_seq = [i2w_char[i] for i in input_seq.numpy()[:l]]
            pred_kvpair = extract_kvpairs_in_bio(pred_seq, char_seq)
            gold_kvpair = extract_kvpairs_in_bio(gold_seq, char_seq)
            preds_kvpair.append(pred_kvpair)
            golds_kvpair.append(gold_kvpair)

    p, r, f1 = cal_f1_score(preds_kvpair, golds_kvpair)
    logger.info("Valid Samples: {}".format(len(preds_kvpair)))
    logger.info("Valid P/R/F1: {} / {} / {}".format(round(p*100, 2), round(r*100, 2), round(f1*100, 2)))

    return (p, r, f1)

epoches = 0
batches = 0
losses = []
best_f1 = 0

while epoches < training_epoches:
    for inputs_seq_batch, inputs_seq_len_batch, outputs_seq_batch in train_dataset:
        if batches == 0: 
            logger.info("###### shape of a batch #######")
            logger.info("input_seq: " + str(inputs_seq_batch.shape))
            logger.info("input_seq_len: " + str(inputs_seq_len_batch.shape))
            logger.info("output_seq: " + str(outputs_seq_batch.shape))
            logger.info("###### preview a sample #######")
            logger.info("input_seq:" + " ".join([i2w_char[i] for i in inputs_seq_batch[0].numpy()]))
            logger.info("input_seq_len :" + str(inputs_seq_len_batch[0]))
            logger.info("output_seq: " + " ".join([i2w_bio[i] for i in outputs_seq_batch[0].numpy()]))
            logger.info("###############################")
        
        with tf.GradientTape() as tape:
            logits_seq = model(inputs_seq_batch, inputs_seq_len_batch, outputs_seq_batch)
            if use_crf:
                log_likelihood, transition_matrix = tfa.text.crf_log_likelihood(logits_seq, outputs_seq_batch, inputs_seq_len_batch, model.transition_params)
                preds_seq, crf_scores = tfa.text.crf_decode(logits_seq, model.transition_params, inputs_seq_len_batch)
                loss = -log_likelihood / tf.cast(inputs_seq_len_batch, tf.float32) # B
            else:
                probs_seq = tf.nn.softmax(logits_seq, axis=-1)
                preds_seq = tf.argmax(probs_seq, axis=-1, name='preds_seq')
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_seq, labels=outputs_seq_batch) # B * S
                masks = tf.sequence_mask(inputs_seq_len_batch, dtype=tf.float32) # B * S
                loss = tf.reduce_sum(loss * masks, axis=-1) / tf.cast(inputs_seq_len_batch, tf.float32) # B
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(loss)
        
        batches += 1
        if (batches + 1) % 10 == 0:
            logger.info("")
            logger.info("Epoches: {}, Batches: {}, Loss: {}".format(epoches, batches, sum(losses) / len(losses)))
            losses = []
            
    p, r, f1 = valid(valid_dataset)
    if f1 > best_f1:
        best_f1 = f1
        save_path = ckpt_manager.save()
        logger.info("Path of ckpt: {}".format(save_path))
        # model = tf.keras.models.load_model(ckpt_save_path)
        logger.info("############# best performance now here ###############")
    epoches += 1
            