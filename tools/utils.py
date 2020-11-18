import numpy as np
import cv2
from config import Config
import tensorflow as tf
from PIL import Image
words_dict = {}
n_2_w = {}
CHARS =set()
with open(Config.dict_file_path, "r", encoding="utf8") as dict_file:
    for index, line in enumerate(dict_file.readlines()):
        line = line.strip()
        if not line == "<BLANK>" and not  line == "<pad>":
            if len(line)!=1:
                print("erro,char's length not equal 1 : {0}".format(line))
                continue
            # assert len(line) == 1, "字典读取错误，存在不是单个字的行"
        words_dict[line] = index
        n_2_w[index] = line
        CHARS.add(line)
Config.dict_size = len(list(CHARS))+2
Config.blank_index = Config.dict_size - 1

def read_img(img_path):
    image = Image.open(img_path)
    img = np.array(image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def resize_norm_img_chinese(img, image_shape):
    des_h,des_w = image_shape
    # imgH, imgW = image_shape
    # todo: change to 0 and modified image shape
    # max_wh_ratio = 0
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h
    # max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(des_h * ratio)
    if imgW > des_w:#如果等比例缩放之后宽度大于了指定的最大宽度
        resize_w = des_w#一般可以丢弃这样的长文本，但是这里我选择了横向缩放，因为我使用的数据集预处理过，不会有特别长的样本出现
    else:
        resize_w = imgW
    resized_img = cv2.resize(img,(resize_w,des_h))
    resized_img = resized_img.astype("float32")
    resized_img = resized_img/255.0
    resized_img -= 0.5
    resized_img /= 0.5
    padding_img = np.zeros((des_h,des_w),dtype=np.float32)
    padding_img[:,:resize_w] = resized_img
    # if math.ceil(imgH * ratio) > imgW:
    #     resized_w = imgW
    # else:
    #     resized_w = int(math.ceil(imgH * ratio))
    # resize_img = cv2.resize(img,(resized_w,imgH))
    # # resized_image = cv2.resize(img, (resized_w, imgH))
    # resized_image = resize_img.astype('float32')
    # # if image_shape[0] == 1:
    # resized_image = resized_image / 255
    # # resized_image = resized_image[np.newaxis, :]
    # # else:
    # #     resized_image = resized_image.transpose((2, 0, 1)) / 255
    # resized_image -= 0.5
    # resized_image /= 0.5
    # padding_im = np.zeros((des_h, des_w), dtype=np.float32)
    # padding_im[:, 0:resized_w] = resized_image
    return padding_img


def transform_img(img):
    '''
    reshape the img and make value to 0~1
    :param img:np.array form
    :return: handled img
    '''
    img_shape = img.shape  # (h,w,c)
    img_h, img_w = img_shape
    if img_h / img_w > 2:
        print("检测到该图片高超过宽的两倍，逆时针旋转90度")
        img = np.rot90(img, 1)
    # if not img_c == 1:
    #     img_c = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = np.squeeze(img)
    img = resize_norm_img_chinese(img, Config.des_img_shape)
    # print(img.shape)
    # img = (img / 255.0 - 0.5) / 0.5
    return img

def ctc_decode(y_pred):
    logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    y_pred = tf.transpose(y_pred,perm=(1,0,2))
    decode_res, logits = tf.nn.ctc_greedy_decoder(y_pred, sequence_length=logit_length)
    # decode_res,logits = tf.nn.ctc_beam_search_decoder(y_pred,sequence_length=logit_length)
    decode_dense = tf.sparse.to_dense(decode_res[0],default_value=0)
    return decode_dense.numpy().tolist()

def map_to_text(tensor):
    # res =[]
    # for item in tensor:
    res = ""
    for item in tensor:
        if item ==  0 or item == Config.blank_index or item == -1:
            continue
        else:
            res+=n_2_w[item]
    # res.append(item_res)
    return res


def cacualte_acc(label,pre):
    # print("label",label)
    # print("pre",pre)
    right_num = 0
    sentence_all_num = label.shape[0]
    # label = [x[:x[-1]] for x in label]
    for index,item in enumerate(pre):
        item_label = label[index].numpy().tolist()
        # if 0 in item_label:
        #     item_label = item_label[:item_label.index(0)]
        # if 0 in item:
        #     item = item[:item.index(0)]
        # print("true: ",item_label)
        # print("pred: ",item)
        pre_text = map_to_text(item)
        label_text = map_to_text(item_label)
        if pre_text == label_text:
            right_num+=1
            # print("right : label_{0},pre_{1}".format(label_text,pre_text))
    return float(right_num/sentence_all_num)

