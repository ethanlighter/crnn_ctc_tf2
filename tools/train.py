import sys
sys.path.append('../')
from tools.utils import ctc_decode,cacualte_acc
from model.model import vgg_crnn
from dataset.tf_data_handler import tf_data_handler
from model.loss import get_tf_ctc_loss
import tensorflow as tf
from tqdm import tqdm
from config import Config
import os
import numpy as np
import logging
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_memory_growth(gpu, True)
# tf_config = tf.ConfigProto()
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 分配50%
# tf_config.gpu_options.allow_growth = True # 自适应
# session = tf.Session(config=tf_config)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# def grad(model,input,targets):
#     with tf.GradientTape() as tape:
#         out = model(input,training = True)
#         loss_value = loss(targets,input)
#     return loss,tape.gradient(loss_value,model.trainable_variables)
# def train():
#     train_loss = []
#     train_acc = []
#     data_handler_obj = data_handler()
#     train_loader = data_handler_obj.get_dataloader(Config.train_anno,Config.img_root)
#     test_loader = data_handler_obj.get_dataloader(Config.test_anno,Config.img_root)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#     summary_writer = tf.summary.create_file_writer(Config.log_dir)
#     model = crnn()
#     model.build(input_shape=(None,32,320,1))
#     step_num = 0
#     with summary_writer.as_default():
#         for epoch in range(Config.epoch):
#             epoch_loss_avg = tf.keras.metrics.Mean()
#             epoch_acc = tf.keras.metrics.Mean()
#             for index,item in enumerate(tqdm(train_loader)):
#                 with tf.GradientTape() as tape:
#                     img_tensor,labels = item
#                     pre_tensor = model(img_tensor,training = True)
#                     loss_value =get_ctc_loss(labels,pre_tensor)
#                     grads = tape.gradient(loss_value,model.trainable_variables)
#                     optimizer.apply_gradients(zip(grads,model.trainable_variables))
#                     epoch_loss_avg(loss_value)
#                 if index%10 == 0:
#                     step_num+=10
#                     item_loss = epoch_loss_avg.result()
#                     print("loss value : epoch_{0} batch_{1} loss_{2}".format(epoch,index,item_loss))
#                     tf.summary.scalar("train-loss",float(item_loss),step=step_num)
#
#             if epoch > 0:
#                 train_acc = eval(model,train_loader)
#                 dev_acc = eval(model,test_loader)
#                 tf.summary.scalar("train-acc",float(train_acc))
#                 tf.summary.scalar("test-acc",float(dev_acc))
#                 model.save(os.path.join(Config.save_model_path,"epoch_{0}_model_acc_{1}".format(epoch,dev_acc)))

def tf_data_train():
    data_handler_obj = tf_data_handler()
    train_loader = data_handler_obj.get_data_loader(Config.train_anno, batch_size=Config.train_batch_size,img_root=Config.img_root,is_train=True)
    test_loader = data_handler_obj.get_data_loader(Config.test_anno, batch_size=Config.train_batch_size,img_root=Config.img_root)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model = vgg_crnn()
    start_epoch = 0
    if len(Config.pre_weight) > 1:
        model.load_weights(Config.pre_weight)
        logging.info("load pretrain weights from {0}".format(Config.pre_weight))
        start_epoch = Config.start_epoch_num
        print("读取模型文件成功")
    summary_writer = tf.summary.create_file_writer(Config.log_dir)
    step_num = 0
    with summary_writer.as_default():
        for epoch in range(start_epoch,Config.epoch):
            epoch_loss_avg = tf.keras.metrics.Mean()
            for index, item in enumerate(tqdm(train_loader)):
                with tf.GradientTape() as tape:
                    img_tensor, labels = item
                    pre_tensor = model(img_tensor, training=True)
                    loss_value = get_tf_ctc_loss(labels, pre_tensor)
                    grads = tape.gradient(loss_value, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    epoch_loss_avg(loss_value)
                if index % 10 == 0:
                    step_num += 10
                    this_loss = epoch_loss_avg.result()
                    tf.summary.scalar("train-loss",float(this_loss),step=step_num)
                    print("loss value : epoch_{0} batch_{1} loss_{2}".format(epoch, index, this_loss))
            if epoch % Config.save_epoch_step == 0:
                print("start test model...")
                # train_acc = eval(model, train_loader)
                dev_acc = eval(model, test_loader)
                # tf.summary.scalar("train-acc", float(train_acc),step=epoch)
                print("test acc : {0}".format(dev_acc))
                tf.summary.scalar("test-acc", float(dev_acc),step=epoch)
                model_save_path = os.path.join(Config.save_model_path, "epoch_{0}_model".format(epoch))
                model.save_weights(model_save_path)
                print("save model to : {0}".format(model_save_path))

def eval(model,test_loader):
    accs = []
    for index,item in enumerate(tqdm(test_loader)):
        img_tensor,labels = item
        pre_tensor = model(img_tensor,training = False)
        pre_index = ctc_decode(pre_tensor)
        accs.append(cacualte_acc(tf.sparse.to_dense(labels),pre_index))
    return np.mean(accs)

if __name__ == "__main__":
    tf_data_train()
