import tensorflow as tf
from tensorflow.keras import layers,models,Model,Sequential

class basic_block(layers.Layer):
    def __init__(self,filter_num,strides = 1):
        super(basic_block, self).__init__()
        self.conv1 = layers.Conv2D(filter_num,(3,3),strides=strides,padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation("relu")
        self.conv2 = layers.Conv2D(filter_num,(3,3),strides=1,padding="same")
        self.bn2 = layers.BatchNormalization()
        if strides != 1:
            self.downsample = layers.Conv2D(filter_num,(1,1),strides=strides)
        else:
            self.downsample = lambda x:x

    def call(self, inputs,training = False):
        out = self.conv1(inputs)
        out = self.bn1(out,training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out,training=training)
        identity = self.downsample(inputs)
        out = layers.add([out,identity])
        out = tf.nn.relu(out)
        return out

class resnet(Model):
    def __init__(self,layers_dim):
        super(resnet, self).__init__()
        self.stem = Sequential([
            layers.Conv2D(64,(3,3),strides=(1,1),padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPool2D(pool_size=(2,2),strides=(2,1),padding="same")
        ])
        self.layer1 = self.build_resblock(64,layers_dim[0],stride = (2,1))
        self.layer2 = self.build_resblock(128,layers_dim[1],stride=(2,1))
        self.layer3 = self.build_resblock(256,layers_dim[2],stride=2)
        self.layer4 = self.build_resblock(512,layers_dim[3],stride=2)
        # self.avgpool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.MaxPool2D(pool_size=(2,2),strides=(1,1))

    def build_resblock(self,filter_num,blocks,stride = (1,1)):
        res_blocks = Sequential()
        res_blocks.add(basic_block(filter_num,strides=stride))
        for _ in range(1,blocks):
            res_blocks.add(basic_block(filter_num,strides=1))
        return res_blocks

    def call(self, inputs, training=None, mask=None):
        x = self.stem(inputs,training = training)
        x = self.layer1(x,training = training)
        x = self.layer2(x,training = training)
        x = self.layer3(x,training = training)
        x = self.layer4(x,training = training)
        # x = self.avgpool(x)
        # x = self.max_pool(x)
        return x


if __name__ == "__main__":
    import numpy as np
    model = resnet(layers_dim=[2,2,2,2])
    img = np.random.random((2, 32, 480, 1))
    tensor = tf.convert_to_tensor(img, tf.float32)
    # model.build(input_shape=(None, 32, 320, 1))
    out = model(tensor, training=True)
    print(out.shape)
    model.summary()