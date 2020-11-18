import argparse
from tools.train import tf_data_train
from tools.test_dev import test
from tools.demo import demo
parser = argparse.ArgumentParser()
parser.add_argument("--type",default="train",help="train or test")
parser.add_argument("--model_file",default=None,help="test model file path")
parser.add_argument("--img_path",default=None,help="test_img_path")
args = parser.parse_args()


def main():
    if args.type == "train":
        tf_data_train()
    elif args.type == "test":
        model_path = args.model_file
        if model_path is None:
            print("请指定测试模型的路径")
            return -1
        test(model_file_path = model_path)
    elif args.type == "demo":
        model_file = args.model_file
        img_path = args.img_path
        if model_file is None or img_path is None:
            print("请指定测试模型和图片所在路径")
            return -1
        demo(model_file,img_path)

if __name__ == "__main__":
    main()
