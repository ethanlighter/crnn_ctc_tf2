
from dataset.tf_data_handler import tf_data_handler
import tensorflow as tf
from config import Config
import time
import os
from model.model import vgg_crnn
from tools.utils import ctc_decode
from tools.utils import map_to_text
import cv2
# tf_config = tf.ConfigProto()
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 分配50%
# tf_config.gpu_options.allow_growth = True # 自适应
# session = tf.Session(config=tf_config)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model_file = "/home/ethony/github_work/crnn_ctc_tf2/checkpoint/epoch_20_model"
model = vgg_crnn()
model.load_weights(model_file)
def demo(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=1)

    img_shape = tf.shape(img)
    scale_factor = Config.des_img_shape[0] / img_shape[0]
    img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
    img_width = tf.cast(img_width, tf.int32)
    img = tf.image.resize(img, (Config.des_img_shape[0], img_width)) / 255.0
    img = tf.expand_dims(img,axis=0)
    pred = model(img)
    pre_index = ctc_decode(pred)
    text = map_to_text(pre_index[0])
    print(text)
if __name__ == "__main__":
    test_path = "/home/ethony/github_work/crnn_ctc_tf2/temp/ture_test_imgs"
    for item in os.listdir(test_path)[:100]:
        if item.endswith("jpg"):
            img_path = os.path.join(test_path,item)
            item_img = cv2.imread(img_path)
            cv2.imshow("item_img",item_img)
            # start_time = time.time()
            # print(img_path)
            demo(img_path)
            cv2.waitKey(0)
            # print(time.time() - start_time)