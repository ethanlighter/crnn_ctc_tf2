import tensorflow as tf
from tensorflow import keras
from config import Config
import numpy as np


def get_tf_ctc_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=None,
        logit_length=logit_length,
        logits_time_major=False,
        blank_index=-1)
    return tf.reduce_mean(loss)

def get_ctc_loss(y_true, y_pred):
    y_true_len = tf.convert_to_tensor([x[-1] for x in y_true], tf.int32)
    y_true = [x[:-1] for x in y_true]
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_true = tf.cast(y_true, tf.int32)
    # y_list = y_true.numpy().tolist()
    # for index,item in enumerate(y_list):
    #     if item == 0:
    #         index
    # this_label_length = tf.convert_to_tensor([np.sum(x) for x in y_true.numpy()])
    # y_true = tf.convert_to_tensor(y_true)
    # this_label_length = tf.fill([tf.shape(y_true)[0]],tf.shape(y_true)[1])

    logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=y_true_len,
        logit_length=logit_length,
        logits_time_major=False,
        blank_index=Config.blank_index)
    return tf.reduce_mean(loss)
class CTCLoss(keras.losses.Loss):
    def __init__(self, logits_time_major=False, blank_index=-1,
                 reduction=keras.losses.Reduction.AUTO, name='ctc_loss'):
        super().__init__(reduction=reduction, name=name)
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index

    def call(self, y_true, y_pred):
        y_true_len = tf.convert_to_tensor([x[-1] for x in y_true],tf.int32)
        y_true = [x[:-1] for x in y_true]
        y_true = tf.convert_to_tensor(y_true,tf.float32)
        y_true = tf.cast(y_true, tf.int32)
        # y_list = y_true.numpy().tolist()
        # for index,item in enumerate(y_list):
        #     if item == 0:
        #         index
        # this_label_length = tf.convert_to_tensor([np.sum(x) for x in y_true.numpy()])
        # y_true = tf.convert_to_tensor(y_true)
        # this_label_length = tf.fill([tf.shape(y_true)[0]],tf.shape(y_true)[1])

        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=y_true_len,
            logit_length=logit_length,
            logits_time_major=False,
            blank_index=Config.blank_index)
        return tf.reduce_mean(loss)