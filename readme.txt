本项目基于tensorflow2.3.1构建了crnn+ctc文字识别模型
crnn模型采用了resnet18+lstm，backbone可选(resnet18或vggnet)
实现功能：
1 不定长文字训练和识别
2 tensorflow内嵌tensorrt加速
3 bacth数据文字识别(待更新)
-----------------------------------------------------------------
参数配置：
config.py
--base_dir 设置为项目所在路径
--train_anno 训练文件路径
--test_anno 测试文件路径
--img_root 图片文件路径
--save_model_path 训练模型保存路径
--pre_weight 预训练文件路径
--dict_file_path 字典文件路径
其它参数可保持不变
------------------------------------------------------------------
训练:
python main.py --type train
测试:
python main.py --type test --model_file {模型参数文件路径}
demo:
python main.py --model_file {模型文件路径} --img_path {图片路径}