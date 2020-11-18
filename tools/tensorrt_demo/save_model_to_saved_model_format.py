from model.model import vgg_crnn
import tensorflow as tf
import numpy as np

model =vgg_crnn()
model.load_weights("/home/ethony/workstation/my_crnn_tf2/checkpoint/epoch_145_model")
model.build(input_shape=(None,32,None,1))
img_tensor = tf.convert_to_tensor(np.random.uniform(-1,1,size=(2,32,320,1)))
pre = model(img_tensor,training = False)
model.save("resnet18_crnn")