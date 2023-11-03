# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/17 21:36
# @author  : Mo
# @function: office of transformers, 训练-主工作流


from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
import torch

from tcLayer import PriorMultiLabelSoftMarginLoss, MultiLabelCircleLoss, LabelSmoothingCrossEntropy, ResampleLoss, FocalLoss, DiceLoss
from tcTools import chinese_extract_extend, mertics_report, sigmoid, softmax
from tcConfig import _TC_MULTI_CLASS, _TC_MULTI_LABEL
from tcGraph import TCGraph as Graph
from tcTqdm import tqdm, trange
from tcAdversarial import FGM

import logging as logger
import numpy as np
import traceback
import random
import codecs
import json
import os


class Office:
    def __init__(self, config, tokenizer, train_corpus=None, dev_corpus=None, tet_corpus=None, logger=logger):
        """
        初始化主训练器/算法网络架构/数据集, init Trainer
        config:
            config: json, params of graph, eg. {"num_labels":17, "model_type":"BERT"}
            train_corpus: List, train corpus of dataset
            dev_corpus: List, dev corpus of dataset 
            tet_corpus: List, tet corpus of dataset 
        Returns:
            None
        """
        self.logger = logger
        self.logger.info(config)
        self.config = config
        self.loss_type = self.config.loss_type if self.config.loss_type else "BCE"
        self.device = "cuda:{}".format(config.CUDA_VISIBLE_DEVICES) if (torch.cuda.is_available() \
            and self.config.is_cuda and self.config.CUDA_VISIBLE_DEVICES != "-1") else "cpu"
        self.logger.info(self.device)
        self.model = Graph(config, tokenizer).to(self.device)  # 初始化模型网络架构
        if self.config.path_finetune:
            self.load_model_state(self.config.path_finetune)
        self.set_random_seed(self.config.seed)  # 初始化随机种子
        self.train_corpus = train_corpus
        self.dev_corpus = dev_corpus
        self.tet_corpus = tet_corpus
        self.prepare_loss()  # 初始化loss等

    def set_random_seed(self, seed):
        """
        初始化随机种子, init seed
        config:
            seed: int, seed of all, eg. 2021
        Returns:
            None
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if self.config.is_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def evaluate(self, mode_type):
        """
        验证, evalate
        config:
            mode_type: str, eg. "tet", "dev"
        Returns:
            result: json
        """
        if mode_type == "test":
            corpus = self.tet_corpus
        elif mode_type == "dev":
            corpus = self.dev_corpus
        else:
            raise Exception("mode_type must be 'dev' or 'tet'")

        data_loader = DataLoader(corpus, batch_size=self.config.batch_size, pin_memory=True)
        ys_pred_id, ys_true_id = None, None
        eval_loss = 0.0
        eval_steps = 0
        self.model.eval()  # 验证
        for batch_data in tqdm(data_loader, desc="evaluate"):
            batch_data = [bd.to(self.device) for bd in batch_data]  # device
            labels = batch_data[3]
            with torch.no_grad():
                inputs = {"attention_mask": batch_data[1],
                          "token_type_ids": batch_data[2],
                          "input_ids": batch_data[0],
                          "labels": labels,
                          }
                logits = self.model(**inputs)
                loss = self.calculate_loss(logits, labels)
                eval_loss += loss.mean().item()
                eval_steps += 1

            inputs_numpy = inputs.get("labels").detach().cpu().numpy()
            logits_numpy = logits.detach().cpu().numpy()
            if ys_pred_id is not None:
                ys_pred_id = np.append(ys_pred_id, logits_numpy, axis=0)
                ys_true_id = np.append(ys_true_id, inputs_numpy, axis=0)
            else:
                ys_pred_id = logits_numpy
                ys_true_id = inputs_numpy
        eval_loss = eval_loss / eval_steps
        # 最后输出, top-1
        ys_true_str, ys_pred_str = [], []
        for i in range(ys_pred_id.shape[0]):
            yti = ys_true_id[i]
            ypi = ys_pred_id[i]
            if _TC_MULTI_LABEL == self.config.task_type.upper():  # 多标签分类, 大于阈值该类别职位1, 否则为0
                ypi[ypi >= self.config.multi_label_threshold] = 1
                ypi[ypi < self.config.multi_label_threshold] = 0
            else:  # 多类分类, 最大得分置为1, 非最大得分全部置为0
                index = ypi.argmax()
                ypi[:] = 0
                ypi[index] = 1
            ys_true_str.append(yti)
            ys_pred_str.append(ypi)
        # 评估
        target_names = [self.config.i2l[str(i)] for i in range(len(self.config.i2l))]
        mertics, report = mertics_report(ys_true_str, ys_pred_str, target_names=target_names)
        self.logger.info(report)
        result = {"loss": eval_loss}
        result.update(mertics)
        return result, report

    def predict(self, corpus, rounded=4, logits_type="logits", use_logits=False):
        """
        预测, pred
        config:
            corpus     : tensor, eg. tensor(1,2,3)
            rounded    : int,   eg. 4
            logits_type: str,   eg. "logits", "sigmoid", "softmax"
        Returns:
            ys_prob: list<json>
        """
        # pin_memory预先加载到cuda-gpu
        data_loader = DataLoader(corpus, batch_size=self.config.batch_size, pin_memory=True)
        ys_pred_id = None
        # 预测 batch-size
        self.model.eval()
        for batch_data in data_loader:
            batch_data = [bd.to(self.device) for bd in batch_data]  # device
            with torch.no_grad():
                inputs = {"attention_mask": batch_data[1],
                          "token_type_ids": batch_data[2],
                          "input_ids": batch_data[0],
                          "labels": batch_data[3],
                          }
                logits = self.model(**inputs)
            logits_numpy = logits.detach().cpu().numpy()
            if logits_type.upper() == "SIGMOID":
                logits_numpy = sigmoid(logits_numpy)
            elif logits_type.upper() == "SOFTMAX":
                logits_numpy = softmax(logits_numpy)
            if ys_pred_id is not None:
                ys_pred_id = np.append(ys_pred_id, logits_numpy, axis=0)
            else:
                ys_pred_id = logits_numpy
        # 只返回onehot形式
        if use_logits:
            return ys_pred_id
        # 最大概率, 最大概率的类别, 类别-概率, like [{"label_1":0.8, "label_2":0.2}]
        ys_prob = []
        for i in range(ys_pred_id.shape[0]):
            ypi = ys_pred_id[i].tolist()
            line = {}
            for idx, prob in enumerate(ypi):
                line[self.config.i2l[str(idx)]] = round(prob, rounded)
            ys_prob.append(line)
        return ys_prob

    def calculate_loss(self, logits, labels):
        """ 计算损失函数 """
        if self.loss_type.upper() ==   "DB_LOSS":
            loss = self.loss_db(logits, labels)
        elif self.loss_type.upper() == "CB_LOSS":
            loss = self.loss_cb(logits, labels)
        elif self.loss_type.upper() == "PRIOR_MARGIN_LOSS":  # 带先验的边缘损失
            loss = self.loss_pmlsm(logits, labels)
        elif self.loss_type.upper() == "SOFT_MARGIN_LOSS": # 边缘损失pytorch版, 划分距离
            loss = self.loss_mlsm(logits, labels)
        elif self.loss_type.upper() == "FOCAL_LOSS":       # 聚焦损失(学习难样本, 2-0.25, 负样本多的情况)
            loss = self.loss_focal(logits.view(-1), labels.view(-1))
        elif self.loss_type.upper() == "CIRCLE_LOSS":      # 圆形损失(均衡, 统一 triplet-loss 和 softmax-ce-loss)
            loss = self.loss_circle(logits, labels)
        elif self.loss_type.upper() == "DICE_LOSS":        # 切块损失(图像)
            loss = self.loss_dice(logits, labels.long())
        elif self.loss_type.upper() == "LABEL_SMOOTH":     # 交叉熵平滑
            loss = self.loss_lsce(logits, labels.long())
        elif self.loss_type.upper() == "BCE_LOGITS":       # 二元交叉熵平滑连续计算型pytorch版
            loss = self.loss_bcelog(logits, labels)
        elif self.loss_type.upper() == "BCE":              # 二元交叉熵的pytorch版, 多类softmax
            logits_softmax = self.softmax(logits)
            loss = self.loss_bce(logits_softmax.view(-1), labels.view(-1))
        elif self.loss_type.upper() == "BCE_MULTI":        # 二元交叉熵的pytorch版, 多标签
            logits_sigmoid = self.sigmoid(logits)
            loss = self.loss_bce(logits_sigmoid.view(-1), labels.view(-1))
        elif self.loss_type.upper() == "MSE":              # 均方误差
            loss = self.loss_mse(logits.view(-1), labels.view(-1))
        elif self.loss_type.upper() == "MIX_focal_prior":  # 混合误差[聚焦损失/2 + 带先验的边缘损失/2]
            loss_focal = self.loss_focal(logits.view(-1), labels.view(-1))
            loss_pmlsm = self.loss_pmlsm(logits, labels)
            loss = (loss_pmlsm + loss_focal) / 2
        elif self.loss_type.upper() == "MIX_focal_prior_9": # 混合误差[聚焦损失4/9 + 带先验的边缘损失5/9]
            loss_focal = self.loss_focal(logits.view(-1), labels.view(-1))
            loss_pmlsm = self.loss_pmlsm(logits, labels)
            loss = (loss_pmlsm*4/9 + loss_focal*5/9)
        elif self.loss_type.upper() == "MIX_focal_bce":   # 混合误差[聚焦损失/2 + 01交叉熵/2]
            loss_focal = self.loss_focal(logits.view(-1), labels.view(-1))
            logits_sigmoid = self.sigmoid(logits)
            loss_bce_multi = self.loss_bce(logits_sigmoid.view(-1), labels.view(-1))
            loss = (loss_bce_multi + loss_focal) / 2
        elif self.loss_type.upper() == "MIX_prior_bce":   # 混合误差[带先验的边缘损失/2 + 01交叉熵/2]
            loss_prior = self.loss_pmlsm(logits, labels)
            logits_sigmoid = self.sigmoid(logits)
            loss_bce_multi = self.loss_bce(logits_sigmoid.view(-1), labels.view(-1))
            loss = (loss_bce_multi + loss_prior) / 2
        else:                                              # 二元交叉熵
            logits_softmax = self.softmax(logits)
            loss = self.loss_bce(logits_softmax.view(-1), labels.view(-1))
        return loss

    def prepare_loss(self):
        """ 准备初始化损失函数 """
        # 损失函数, cb_loss/db_loss能提高验证指标,但鲁棒性等大概率会降低
        self.loss_cb = ResampleLoss(reweight_func="CB", loss_weight=10.0,
                              focal=dict(focal=True, alpha=0.5, gamma=2),
                              logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                              CB_loss=dict(CB_beta=0.9, CB_mode="by_class"),
                              class_freq=self.config.prior_count,
                              train_num=self.config.len_corpus)

        self.loss_db = ResampleLoss(reweight_func="rebalance", loss_weight=1.0,
                              focal=dict(focal=True, alpha=0.5, gamma=2),
                              logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                              map_param=dict(alpha=0.1, beta=10.0, gamma=0.9),
                              class_freq=self.config.prior_count,
                              train_num=self.config.len_corpus)

        self.loss_pmlsm = PriorMultiLabelSoftMarginLoss(prior=self.config.prior, num_labels=self.config.num_labels)
        self.loss_lsce = LabelSmoothingCrossEntropy()
        self.loss_circle = MultiLabelCircleLoss()
        self.loss_focal = FocalLoss()
        self.loss_dice = DiceLoss()

        self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.loss_mlsm = torch.nn.MultiLabelSoftMarginLoss()  # like BCEWithLogitsLoss
        self.loss_bcelog = torch.nn.BCEWithLogitsLoss()
        self.loss_bce = torch.nn.BCELoss()
        self.loss_mse = torch.nn.MSELoss()
        # 激活层/随即失活层
        self.softmax = torch.nn.Softmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()

    def train_model(self):
        """  训练迭代epoch
        return global_steps, best_mertics
        """
        #  数据转为迭代器iter的形式    #  weight_decay_rate=0.01, grad_accum_steps
        data_loader = DataLoader(self.train_corpus, sampler=RandomSampler(self.train_corpus), batch_size=self.config.batch_size, pin_memory=True)

        #  配置好优化器与训练工作计划(主要是学习率预热warm-up与衰减weight-decay, 以及不作用的层超参)
        params_no_decay = ["LayerNorm.weight", "bias"]
        parameters_no_decay = [
            {"params": [p for n, p in self.model.named_parameters() if not any(pnd in n for pnd in params_no_decay)],
             "weight_decay": self.config.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(pnd in n for pnd in params_no_decay)],
             "weight_decay": 0.0}
            ]
        optimizer = AdamW(parameters_no_decay, lr=self.config.lr, eps=self.config.adam_eps)
        # 训练轮次
        times_batch_size = len(data_loader) // self.config.grad_accum_steps
        num_training_steps = int(times_batch_size * self.config.epochs)
        # 如果选择-1不设置则为 半个epoch
        num_warmup_steps = int((len(data_loader) // self.config.grad_accum_steps // 2)) if self.config.warmup_steps == -1 else self.config.warmup_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        tensorboardx_witer = SummaryWriter(logdir=self.config.model_save_path)

        # adv
        if self.config.is_adv:
            fgm = FGM(self.model, emb_name=self.config.adv_emb_name, epsilon=self.config.adv_eps)
        # 开始训练
        self.model.graph_config.is_train = True
        epochs_store = []
        global_steps = 0
        best_mertics = {}
        best_report = ""
        for epochs_i in trange(self.config.epochs, desc="epoch"):  # epoch
            self.model.train()  # train-type
            for idx, batch_data in enumerate(tqdm(data_loader, desc="epoch_{} step".format(epochs_i))):  # step
                # 数据与模型
                batch_data = [bd.to(self.device) for bd in batch_data]  # device
                labels = batch_data[3]
                inputs = {"attention_mask": batch_data[1],
                          "token_type_ids": batch_data[2],
                          "input_ids": batch_data[0],
                          "labels": labels,
                          }
                logits = self.model(**inputs)
                loss = self.calculate_loss(logits, labels)
                loss = loss / self.config.grad_accum_steps
                loss.backward()
                global_steps += 1
                #  对抗训练
                if self.config.is_adv:
                    fgm.attack()  # 在embedding上添加对抗扰动
                    logits = self.model(**inputs)
                    loss = self.calculate_loss(logits, labels)
                    loss = loss / self.config.grad_accum_steps  # 梯度累计
                    loss.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore()  # 恢复embedding参数
                #  梯度累计
                if (idx + 1) % self.config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                # 评估算法/打印日志/存储模型, 1个epoch/到达保存的步数/或者是最后一轮最后一步
                if (self.config.evaluate_steps > 0 and global_steps % self.config.evaluate_steps == 0) or (idx == times_batch_size-1) \
                        or (epochs_i+1 == self.config.epochs and idx+1 == len(data_loader)):
                    res, report = self.evaluate("dev")
                    self.logger.info("best_report\n" + best_report)
                    self.logger.info("current_mertics: {}".format(res))
                    self.logger.info("epoch_global: {}, step_global: {}, step: {}".format(epochs_i, global_steps, idx))
                    # idx_score = res.get("micro", {}).get("f1", 0)  # "macro", "micro", "weighted"
                    for k,v in res.items():  # tensorboard日志, 其中抽取中文、数字和英文, 避免一些不支持的符号, 比如说 /\|等特殊字符
                        if type(v) == dict:  # 空格和一些特殊字符tensorboardx.add_scalar不支持
                            k = chinese_extract_extend(k)
                            k = k.replace(" ", "")
                            for k_2, v_2 in v.items():
                                tensorboardx_witer.add_scalar(k + "/" + k_2, v_2, global_steps)
                        elif type(v) == float:
                            tensorboardx_witer.add_scalar(k, v, global_steps)
                            tensorboardx_witer.add_scalar("lr", scheduler.get_lr()[-1], global_steps)
                    self.model.train()  # 预测时候的, 回转回来
                    save_best_mertics_key = self.config.save_best_mertics_key  # 模型存储的判别指标
                    abmk_1 = save_best_mertics_key[0]  # like "micro_avg"
                    abmk_2 = save_best_mertics_key[1]  # like "f1-score"
                    if res.get(abmk_1, {}).get(abmk_2, 0) > best_mertics.get(abmk_1, {}).get(abmk_2, 0):  # 只保留最优的指标
                        epochs_store.append((epochs_i, idx))
                        res["total"] = {"epochs": epochs_i, "global_steps": global_steps, "step_current": idx}
                        best_mertics = res
                        best_report = report
                        self.save_model_state()
                    # 早停, 连续stop_epochs轮指标不增长则自动停止
                    if epochs_store and epochs_i - epochs_store[-1][0] >= self.config.stop_epochs:
                        break
        return global_steps, best_mertics, best_report

    def load_model_state(self, path_dir=""):
        """  仅加载模型参数(推荐使用)  """
        try:
            if path_dir:
                path_model = path_dir
            else:
                path_model = os.path.join(self.config.model_save_path, self.config.model_name)
            self.model.load_state_dict(torch.load(path_model, map_location=torch.device(self.device)))
            self.model.to(self.device)
            self.logger.info("******model loaded success******")
            self.logger.info("self.device: {}".format(self.device))
        except Exception as e:
            self.logger.info(str(traceback.print_exc()))
            raise Exception("******load model error******")

    def save_model_state(self):
        """  仅保存模型参数(推荐使用)  """
        if not os.path.exists(self.config.model_save_path):
            os.makedirs(self.config.model_save_path)
        # save config
        self.config.is_train = False  # 存储后的用于预测
        path_config = os.path.join(self.config.model_save_path, self.config.config_name)
        with codecs.open(filename=path_config, mode="w", encoding="utf-8") as fc:
            json.dump(vars(self.config), fc, indent=4, ensure_ascii=False)
            fc.close()
        # save model
        path_model = os.path.join(self.config.model_save_path, self.config.model_name)
        torch.save(self.model.state_dict(), path_model)
        self.logger.info("******model_save_path is {}******".format(path_model))

    def save_onnx(self, path_onnx_dir=""):
        """  存储为ONNX格式  """
        ### ONNX---路径
        if not path_onnx_dir:
            path_onnx_dir = os.path.join(self.config.model_save_path, "onnx")
        if not os.path.exists(path_onnx_dir):
            os.makedirs(path_onnx_dir)
        batch_data = [[[1, 2, 3, 4]*32]*32, [[1, 0]*64]*32, [[0, 1]*64]*32]
        # for name, param in self.model.named_parameters():  # 查看可优化的参数有哪些
        #     # if param.requires_grad:
        #         print(name)
        # batch_data = [bd.to(self.device) for bd in batch_data]  # device
        with torch.no_grad():
            inputs = {"attention_mask": torch.tensor(batch_data[1]).to(self.device),
                      "token_type_ids": torch.tensor(batch_data[2]).to(self.device),
                      "input_ids": torch.tensor(batch_data[0]).to(self.device),
                      }
            _ = self.model(**inputs)
            input_names = ["input_ids", "attention_mask", "token_type_ids"]
            output_names = ["outputs"]
            torch.onnx.export(self.model,
            (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
            os.path.join(path_onnx_dir, "tc_model.onnx"),
            input_names=input_names,
            output_names=output_names,  ## Be carefule to write this names
            opset_version=10,  # 7,8,9
            do_constant_folding=True,
            use_external_data_format=True,
            dynamic_axes = {
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                output_names[0]: {0: "batch"}
            })

    def load_model(self, path_dir=""):
        """  加载模型  """
        try:
            if path_dir:
                path_model = path_dir
            else:
                path_model = os.path.join(self.config.model_save_path, self.config.model_name)
            self.model = torch.load(path_model, map_location=torch.device(self.device))
            self.logger.info("******model loaded success******")
        except Exception as e:
            self.logger.info(str(traceback.print_exc()))
            raise Exception("******load model error******")

    def save_model(self):
        """  存储模型  """
        if not os.path.exists(self.config.model_save_path):
            os.makedirs(self.config.model_save_path)
        # save config
        self.config.is_train = False  # 存储后的用于预测
        path_config = os.path.join(self.config.model_save_path, self.config.config_name)
        with codecs.open(filename=path_config, mode="w", encoding="utf-8") as fc:
            json.dump(vars(self.config), fc, indent=4, ensure_ascii=False)
            fc.close()
        # save model
        path_model = os.path.join(self.config.model_save_path, self.config.model_name)
        torch.save(self.model, path_model)
        self.logger.info("******model_save_path is {}******".format(path_model))
