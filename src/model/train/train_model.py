"""
    This code is to train model using pytorch_nlu.

"""
# add path to python path
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import argparse
# import nlu functions
from pytorch_nlu.pytorch_sequencelabeling.slTools import get_current_time
from pytorch_nlu.pytorch_sequencelabeling.slRun import SequenceLabeling
from pytorch_nlu.pytorch_sequencelabeling.slConfig import model_config

# config model:
MODEL_TYPE = ["BERT", "ERNIE", "BERT_WWM", "ALBERT", "ROBERTA", "XLNET", "ELECTRA"]
# now we only use BERT as our base model
PRETRAINED_MODEL_NAME_OR_PATH = {
    "BERT_WWM":  "", 
    "ROBERTA":  "", 
    "ALBERT":  "", 
    "XLNET":  "", 
    "ERNIE":  "", 
    "BERT":  "src/model/pretrained_model/bert-base-chinese",
    # "BERT":  "src/model/pretrained_model/bert-base-uncased",
}

# init all config
argparser = argparse.ArgumentParser()
# for dataset
argparser.add_argument('--dataset_path', type=str, default='src/model/dataset/lll_dataset_v1',
                          help='path of dataset')
# for model
argparser.add_argument('--model_type', type=str, default='BERT', choices=MODEL_TYPE,
                       help='type of model')
# for train
argparser.add_argument('--batch_size', type=int, default=32,
                          help='size of each batch')
argparser.add_argument('--epoch', type=int, default=50,
                            help='number of epoches')
argparser.add_argument('--learning_rate', type=float, default=0.0001,
                            help='learning rate')
argparser.add_argument('--dense_lr', type=float, default=0.0001,
                            help='learning rate of dense layer')
argparser.add_argument('--max_len', type=int, default=256,
                            help='max length of sentence')
# for result
argparser.add_argument('--result_path', type=str, default='src/model/train_result',
                          help='path of result')
cfg = argparser.parse_args()

# create result_path
result_path = os.path.join(cfg.result_path, cfg.model_type, "{}_{}".format(cfg.model_type, get_current_time()))
if not os.path.exists(result_path):
    os.makedirs(result_path)

# auto get dataset path
assert os.path.exists(cfg.dataset_path), "dataset path not exist"
# get dataset file in dataset path with postfix with "train.txt/dev.txt/test.txt"
dataset_file = [os.path.join(cfg.dataset_path, file) for file in os.listdir(cfg.dataset_path) if file.endswith(".txt")]
# get and check train/dev/test file
train_file = [file for file in dataset_file if "train" in file]
assert len(train_file) == 1, "train file not exist or more than one"
train_file = train_file[0]
dev_file = [file for file in dataset_file if "dev" in file]
assert len(dev_file) == 1, "dev file not exist or more than one"
dev_file = dev_file[0]
test_file = [file for file in dataset_file if "test" in file]
assert len(test_file) == 1, "test file not exist or more than one"
test_file = test_file[0]

# set dataset config
model_config["path_train"] = train_file
model_config["path_dev"] = dev_file
model_config["path_tet"] = test_file
model_config["corpus_type"] = "DATA-SPAN" # annotation format "DATA-CONLL" or "DATA-SPAN"
model_config["task_type"] = "SL-SOFTMAX"

# train config
model_config["batch_size"] = cfg.batch_size
model_config["epochs"] = cfg.epoch
model_config["lr"] = cfg.learning_rate
model_config["dense_lr"] = cfg.dense_lr
model_config["max_len"] = cfg.max_len

# set model config
model_config["pretrained_model_name_or_path"] = PRETRAINED_MODEL_NAME_OR_PATH[cfg.model_type]
model_config["model_type"] = cfg.model_type
model_config["model_save_path"] = result_path

# main
lc = SequenceLabeling(model_config)
lc.process()
lc.train()


# shell
# nohup python src/model/train/train_model.py > train.log 2>&1 &