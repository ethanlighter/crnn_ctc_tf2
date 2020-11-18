
from dataset.tf_data_handler import tf_data_handler
import tensorflow as tf
from config import Config
import time
from model.model import vgg_crnn
from tools.utils import ctc_decode
from tools.utils import map_to_text
# tf_config = tf.ConfigProto()
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 分配50%
# tf_config.gpu_options.allow_growth = True # 自适应
# session = tf.Session(config=tf_config)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def demo(model_file_path,img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=1)

    img_shape = tf.shape(img)
    scale_factor = Config.des_img_shape[0] / img_shape[0]
    img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
    img_width = tf.cast(img_width, tf.int32)
    img = tf.image.resize(img, (Config.des_img_shape[0], img_width)) / 255.0
    img = tf.expand_dims(img,axis=0)
    model = vgg_crnn()
    model.load_weights(model_file_path)
    pred = model(img)
    pre_index = ctc_decode(pred)
    text = map_to_text(pre_index[0])
    print(text)
if __name__ == "__main__":
    model_file = "/home/ethony/workstation/my_crnn_tf2/checkpoint/200W/epoch_23_model"
    img = "/home/ethony/workstation/my_crnn_tf2/dataset/test_imgs_01/0.jpg"
    demo(model_file,img)