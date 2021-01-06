import os
class Config():
    base_dir = os.path.dirname(__file__)
    des_img_shape = (32,320)
    dict_file_path = os.path.join(base_dir,"dicts","ppocr_keys_v1.txt")
    dict_size = None
    max_seq_length = 48
    train_batch_size = 32
    max_time_step = 2*max_seq_length+1
    resnet = "ResNet18"#支持vgg和resnet18
    lstm_units = 128
    save_epoch_step = 1
    blank_index = None
    is_abandon_long_imgs = False  #是否丢弃过长样本，建议在数据预处理环节去掉，否在每次运行都会处理一遍占用时间
    log_dir = os.path.join(base_dir,"log")
    epoch=150
    train_anno = "/home/ethony/github_work/crnn_ctc_tf2/temp/select_train.txt"
    test_anno = "/home/ethony/github_work/crnn_ctc_tf2/temp/select_test.txt"
    img_root = "/"
    save_model_path = "/home/ethony/github_work/crnn_ctc_tf2/checkpoint"
    pre_weight = "/home/ethony/github_work/crnn_ctc_tf2/checkpoint/epoch_16_model"
    start_epoch_num = 16