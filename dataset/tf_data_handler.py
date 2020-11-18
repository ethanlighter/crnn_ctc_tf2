import tensorflow as tf
from tqdm import tqdm
from tools.utils import transform_img,read_img
from config import Config
import os
from PIL import Image
class tf_data_handler():
    def __init__(self):
        self.table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            Config.dict_file_path, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER), -1)
        self.w2id = {}
        self.id2w = {}
        with open(Config.dict_file_path,"r",encoding="utf8") as dict_file:
            for index,line in enumerate(dict_file):
                self.w2id[line.strip()] = index
                self.id2w[index] = line.strip()

    def read_file_from_local(self, anno_path, img_root):
        img_paths = []
        labels = []
        with open(anno_path, "r", encoding="utf8") as train_file:
            print("读取anno文件...")
            for line in tqdm(train_file.readlines()):
                line_split = line.split("\t")
                assert len(line_split) == 2, "数据文件格式错误，标签文件每一行的格式应该为 img_relative_path\timg_label\n"
                img_path = line_split[0].strip()
                img_label = line_split[1].strip()
                if len(img_label) > Config.max_seq_length:#如果文本的长度超过了最大长度限制，丢弃样本
                    continue
                img_path = os.path.join(img_root,img_path)
                #--------------丢弃宽度过长的样本，如果训练数据大的会占用大量时间，建议写一个预处理的脚本替换
                if Config.is_abandon_long_imgs:
                    image = Image.open(img_path)
                    if image.size[0] > Config.max_time_step*4:
                        continue
                # ---------------
                img_paths.append(img_path)
                labels.append(img_label)

        return img_paths,labels

    def _decode_img(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=1)

        img_shape = tf.shape(img)
        scale_factor = Config.des_img_shape[0] / img_shape[0]
        img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
        img_width = tf.cast(img_width, tf.int32)
        img = tf.image.resize(img, (Config.des_img_shape[0], img_width)) / 255.0
        return img, label
    # def _handle_img_path(self,img_path,label):
    #     img = read_img(img_path)
    #     img = transform_img(img)
    #     img = tf.convert_to_tensor(img,tf.float32)
    #     return img,label
    # def _handle_label(self,img,lable):
    #     label_tensor = []
    #     for index,c in enumerate(lable):
    #         label_tensor.append(self.w2id[c])
    #     label_tensor = tf.convert_to_tensor(label_tensor,tf.int32)
    #     return img,label_tensor
    def _tokenize(self, imgs, labels):
        # print(labels)
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = tf.ragged.map_flat_values(self.table.lookup, chars)
        tokens = tokens.to_sparse()

        return imgs, tokens
    def get_data_loader(self,anno_path,batch_size,img_root,is_train = False):
        img_paths,img_labels = self.read_file_from_local(anno_path,img_root)
        ds = tf.data.Dataset.from_tensor_slices((img_paths,img_labels))
        if is_train:
            ds = ds.shuffle(buffer_size=10000)
        ds = ds.map(self._decode_img,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # ds = ds.map(lambda x,y : tf.py_function(self._handle_img_path,inp=[x,y],Tout=tf.float32),num_parallel_calls=tf.data.experimental.AUTOTUNE)#num_parallel_calls 表示并行读取文件的数量
        if batch_size > 1:
            ds = ds.padded_batch(batch_size, drop_remainder=is_train)
        # ds = ds.apply(tf.data.experimental.ignore_errors())
        ds = ds.map(self._tokenize,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # ds = ds.map(lambda x,y : tf.py_function(self._handle_label,inp=[x,y],Tout=tf.int32),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

if __name__ == "__main__":
    handler_obj = tf_data_handler()
    train_ds = handler_obj.get_data_loader("/home/ethony/workstation/my_crnn_tf2/dataset/datas.txt",16,Config.img_root)
    print(train_ds)
    # batchs = [x for x in iter(train_ds)]
    # a = batchs[0]
    # b = a[1]
    # print(b)
    for item in iter(train_ds):
        img_tensor,label_sparse = item
        print(img_tensor.shape," ",label_sparse.shape)
        # print(label_sparse.shape)

    # print(tf.sparse.to_dense(b))
    # for index,item in enumerate(dataloader.as_numpy_iterator()):
    #     print(item)



