import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import time
import numpy as np

saved_model = tf.saved_model.load("resnet18_crnn_trt",tags=[trt.tag_constants.SERVING])
graph_func = saved_model.signatures[
trt.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
]
frozen_func = trt.convert_to_constants.convert_variables_to_constants_v2(
    graph_func)
img_tensor = tf.convert_to_tensor(np.random.uniform(-1,1,size=(2,32,320,1)),dtype = tf.float32)
for i in range(10):
    start_time = time.time()
    pred = frozen_func(img_tensor)
    print("cost time : {0}".format(time.time() - start_time))


