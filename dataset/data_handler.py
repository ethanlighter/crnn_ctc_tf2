import cv2
from PIL import Image
import numpy as np
from config import Config
import os
import tensorflow as tf
from tqdm import tqdm
import random

from tools.utils import transform_img
class dataloader():
    def __init__(self,samples,shuffle = False,batch_Size=Config.train_batch_size):
        self.samples = samples
        self.start_index = 0
        self.batch_size = batch_Size
        if shuffle:
            random.shuffle(self.samples)
        self.tensor_data = self.convert_samples_to_tensor(self.samples)
    def convert_text_to_sparse_tensor(self,text):
        this_dense_shape = [Config.max_time_step,Config.dict_size]
        this_indice = []
        this_values = []
        for index,c_index in enumerate(text):
            this_indice.append([index,c_index])
            this_values.append(1)
        this_sparse_tensor = tf.sparse.SparseTensor(
            indices=this_indice,
            values = this_values,
            dense_shape=this_dense_shape
        )
        return this_sparse_tensor
    # def convert_text_to_tensor(self,text):
    #     result = [[0]*Config.dict_size]*Config.max_seq_length
    #     for index,c_index in enumerate(text):
    #         result[index][c_index] = 1
    #     # result = tf.convert_to_tensor(result,tf.float32)
    #     return result
    def convert_text_to_tensor(self,text,max_seq_length):
        result = [0]*max_seq_length
        for index,c_index in enumerate(text):
            result[index] = c_index
        result.append(len(text))
        # result = tf.convert_to_tensor(result,tf.float32)
        return result

    def convert_samples_to_tensor(self,samples):
        max_width = 0
        max_seq_length = 0
        imgs = []
        labels = []
        text_lengths = []
        for item in samples:
            img,label = item
            h,w = img.shape
            seq_length = len(label)
            if w > max_width:
                max_width = w
            if seq_length>max_seq_length:
                max_seq_length = seq_length
        for item in samples:
            img,label = item
            h,w = img.shape
            if w<max_width:
                temp = np.zeros((Config.des_img_shape[0],max_width),np.float)
                temp[:,:w] = img
                img = temp
            # text_len = len(label)
            label = self.convert_text_to_tensor(label,max_seq_length)
            # if seq_length < max_seq_length:
            #     temp = np.zeros((max_seq_length,Config.dict_size),np.int)
            #     temp[:seq_length,:] = label
            #     label = temp
            img = img[:,:,np.newaxis]
            imgs.append(img)
            labels.append(label)
            # text_lengths.append(text_len)
        img_tensor = tf.convert_to_tensor(imgs,tf.float32)
        # labels_tensor = tf.convert_to_tensor(labels,tf.float32)
        # len_tensor = tf.convert_to_tensor(text_lengths,tf.int32)
        # label_tensor = tf.convert_to_tensor(labels,tf.int16)
        return (img_tensor,labels)

    # def get_next_batch(self):
    #     this_samples = []
    #     sample_length = len(self.samples)
    #     if self.start_index+self.batch_size > len(self.samples):
    #         this_samples.extend(self.samples[self.start_index:])
    #         self.start_index = self.batch_size - (len(self.samples) - self.start_index)
    #         this_samples.extend(self.samples[:self.start_index])
    #     else:
    #         this_samples.extend(self.samples[self.start_index:self.start_index+self.batch_size])
    #         self.start_index+=self.batch_size
    #     tensor_data = self.convert_samples_to_tensor(this_samples)
    #     return tensor_data
    def get_next_batch(self):
        this_img_tensor = []
        this_label = []
        tensors_length = len(self.tensor_data)
        if self.start_index+self.batch_size > tensors_length:
            this_img_tensor.extend(self.tensor_data[0][self.start_index:])
            this_label.extend(self.tensor_data[1][self.start_index:])
            self.start_index = self.batch_size - (tensors_length - self.start_index)
            this_img_tensor.extend(self.tensor_data[0][:self.start_index])
            this_label.extend(self.tensor_data[1][:self.start_index])
        else:
            this_img_tensor.extend(self.tensor_data[0][self.start_index:self.start_index+self.batch_size])
            this_label.extend(self.tensor_data[1][self.start_index:self.start_index+self.batch_size])
            self.start_index+=self.batch_size
        return this_img_tensor,this_label
    def __iter__(self):
        return self
    def __next__(self):
        return self.get_next_batch()
    def __str__(self):
        return "数据迭代器，总数据长度: {0}".format(len(self.samples))







class data_handler():
    def __init__(self):
        self.train_datas = []
        self.test_datas = []
        self.words_dict = {}
        with open(Config.dict_file_path,"r",encoding="utf8") as dict_file:
            for index,line in enumerate(dict_file.readlines()):
                line = line.strip()
                if not line == "<BLANK>":
                    assert len(line)==1,"字典读取错误，存在不是单个字的行"
                self.words_dict[line] = index
        self.dict_length = len(self.words_dict.items())

    def handle_label(self, sentence):
        if not Config.use_space:
            sentence = sentence.replace(" ", "")
        # result = [[0] * (self.dict_length)] * len(sentence)
        result = []
        for index, c in enumerate(sentence):
            if c in self.words_dict:
                c_index = self.words_dict[c]
                result.append(c_index)
            else:
                print("注意存在字典中没有的字，该样本抛弃:{0}".format(sentence))
                return None
        assert len(result) == len(sentence)
        return result
    def read_img(self,img_path):
        image = Image.open(img_path)
        img = np.array(image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return img

    def read_file_from_local(self,anno_path,img_root):
        datas = []
        with open(anno_path,"r",encoding="utf8") as train_file:
            print("读取anno文件...")
            for line in tqdm(train_file.readlines()):
                line_split = line.split("\t")
                assert len(line_split) == 2,"数据文件格式错误，标签文件每一行的格式应该为 img_relative_path\timg_label\n"
                img_path = line_split[0].strip()
                img_label = line_split[1].strip()
                datas.append((os.path.join(img_root,img_path),img_label))
        return datas
    def convert_file_to_sample(self,datas):

        print("convert data ....")
        # imgs = []
        # labels = []
        samples = []
        for item in tqdm(datas):
            img_path,img_label = item
            img = self.read_img(img_path)
            img = transform_img(img)
            label = self.handle_label(img_label)
            if not label is None:
                # imgs.append(img)
                # labels.append(labels)
                samples.append((img,label))
        return samples
    def get_dataloader(self,anno_path,img_root):
        datas = self.read_file_from_local(anno_path,img_root)
        samples = self.convert_file_to_sample(datas)
        data_loader = dataloader(samples,shuffle=True)
        return data_loader

if __name__ == "__main__":
    data_hadler_obj = data_handler()
    train_dataloader = data_hadler_obj.get_dataloader(Config.train_anno,Config.img_root)
    for index,item in enumerate(train_dataloader):
        img_tensor,label = item
        print(img_tensor.shape," ",label)









 