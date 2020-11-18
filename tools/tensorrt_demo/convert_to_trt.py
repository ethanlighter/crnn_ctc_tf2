
from dataset.tf_data_handler import tf_data_handler
import tensorflow as tf
from config import Config
import time
from model.model import vgg_crnn
import numpy as np
from tools.utils import ctc_decode,cacualte_acc


data_handler_obj = tf_data_handler()
test_loader = data_handler_obj.get_data_loader(Config.test_anno, batch_size=Config.train_batch_size,img_root=Config.img_root)
# model = crnn()
# model = build_model(Config.dict_size)
model = vgg_crnn()
# model.build(input_shape=(None,32,None,1))
model.load_weights("/home/ethony/workstation/my_crnn_tf2/checkpoint/200W/epoch_0_model")
# model_path = "/home/ethony/workstation/my_crnn_tf2/checkpoint/epoch_0_model"
# model = tf.keras.models.load_model(model_path)
# model.build(input_shape=(None, 32, None, 1))
img_tensor = tf.convert_to_tensor(np.random.uniform(-1,1,size=(2,32,320,1)),dtype = tf.float32)
for i in range(10):
    start_time = time.time()
    pred = model(img_tensor)
    print("cost time : {0}".format(time.time() - start_time))

