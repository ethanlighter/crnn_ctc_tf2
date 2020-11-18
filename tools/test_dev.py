from dataset.tf_data_handler import tf_data_handler
import tensorflow as tf
from config import Config
import time
from model.model import vgg_crnn
from tools.utils import ctc_decode,cacualte_acc
import numpy as np

# tf_config = tf.ConfigProto()
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 分配50%
# tf_config.gpu_options.allow_growth = True # 自适应
# session = tf.Session(config=tf_config)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def test(model_file_path):
    data_handler_obj = tf_data_handler()
    test_loader = data_handler_obj.get_data_loader(Config.test_anno, batch_size=Config.train_batch_size,img_root=Config.img_root)
    model = vgg_crnn()
    model.load_weights(model_file_path)
    accs = []
    for index,item in enumerate(test_loader):
        img_tensor,labels = item
        print(img_tensor.shape)
        start_time = time.time()
        pre_tensor = model(img_tensor)
        # print(pre_tensor.shape)
        print("cost time : {0}".format(time.time() - start_time))
        pre_index = ctc_decode(pre_tensor)
        # print(pre_index)
        acc = cacualte_acc(tf.sparse.to_dense(labels),pre_index)
        accs.append(acc)
    print(np.mean(accs))

if __name__ == "__main__":
    test("/home/ethony/workstation/my_crnn_tf2/checkpoint/200W/epoch_23_model")

