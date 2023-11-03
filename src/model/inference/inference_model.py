"""
    This file is disigned for trained model load and inference
"""
import os
import sys
# add path to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from pytorch_nlu.pytorch_sequencelabeling.slPredict import SequenceLabelingPredict


if __name__ == "__main__":
    path_config = "./src/model/train_result/BERT/BERT_20231103101604/sl.config"
    tcp = SequenceLabelingPredict(path_config)
    texts = [
        {"text": "[愛戀字幕社][10月新番][星靈感應][hoshikuzu telepath][03][1080p][mp4][big5][繁中]"},
                ]
    res = tcp.predict(texts)
    print(res)
    while True:
        print("请输入:")
        question = input()
        res = tcp.predict([{"text": question}])
        print(res)

# how to use:
# python src/model/inference/inference_model.py