
from __future__ import division
import os
import pickle
import torch
import time
import json
import logging
import sys
import re
import pandas as pd
import shutil
import torch.nn as nn
import torch.nn.functional as F
from math import pi, cos
import numpy as np
from sklearn import linear_model
import copy

class IOdata:
    def __init__(self, root, root_load, save_name = "", start_epoch = 0):
        if save_name is "":
            localtime = time.localtime(time.time())
            save_name = "result_"+"_".join([str(x_) for x_ in list(localtime)[:-3]])
        else:
            save_name = save_name
        self.root = os.path.join(root, save_name)
        print(f'================={self.root}======================')
        os.makedirs(self.root, exist_ok=True)
        logging.basicConfig(format='%(asctime)s - '+save_name+': %(message)s', # Format for basic line.
                            level=logging.INFO,
                            filename=os.path.join(self.root, 'out.log'),
                            filemode='a', 
                            datefmt='%Y-%m-%d %H:%M:%S') # The format of data
        ch =logging.StreamHandler()
        ch.setLevel(logging.INFO)
        self.logger = logging.getLogger()
        self.logger.addHandler(ch)
        self.root_load = root_load
    def save(self, data_name_str, subname = None): # Saving all the document, including the csv, pkl and pth.
        local = sys._getframe().f_back.f_locals
        #print(local)
        data_name_list = [x_ for x_ in re.split(' |,', data_name_str) if len(x_)>0]
        data_list = [local[x_] for x_ in data_name_list] # Turn name into real function object.
        if isinstance(subname, int):
            subname = "epoch_"+str(subname)
        save_folder = os.path.join(self.root, subname, '')
        os.makedirs(save_folder, exist_ok=True)
        self.Uid = -1
        for key, data in zip(data_name_list, data_list):
            if hasattr(data, 'state_dict'): #assume it is a model
                torch.save(data.state_dict(), save_folder+key+".pth.tar")
            elif isinstance(data, pd.core.frame.DataFrame):
                data.to_csv(save_folder+key+'.csv', sep='\t', encoding='utf-8')
            elif torch.is_tensor(data):
                torch.save(data, save_folder+key+'.pt')
            else:
                with open(save_folder+key+'.pkl', 'wb') as fid:
                    pickle.dump(data, fid, protocol=pickle.HIGHEST_PROTOCOL)
    # Loading several models at once.
    def load(self, data_name_list, save_folder):
        if isinstance(save_folder, int) or isinstance(save_folder, str):
            save_folder = os.path.join("pretrain", self.root_load, "epoch_"+str(save_folder))
        print('---------Cloud Load---------', save_folder)
        file_list = os.listdir(save_folder)
        out = dict()
        for data_name in data_name_list:
            file_name = self.in_start(data_name+".", file_list) # Get the file starting with data_name
            if file_name == '':
                print(file_name+" is not found in "+save_folder)
            if file_name[-8:] == ".pth.tar" or 'yaml' in file_name: #assume it is a model
                out[data_name] = torch.load(os.path.join(save_folder, file_name))
            else: # In case of dictionary
                with open(os.path.join(save_folder, file_name), 'rb') as fid:
                    out[data_name] = pickle.load(fid)
        if len(out) == 1:
            return out[data_name]
        else:
            return (*[out[data_name] for data_name in data_name_list], )
    def get_variable_name(self, variable, id2loc):
        var_id = id(variable)
        if var_id in id2loc.keys():
            return id2loc[var_id]
        else:
            self.Uid += 1
            return 'Unknown_Var '+str(self.Uid)
    def in_start(self, short_str, strlist):
        for long_str in strlist:
            if short_str == long_str[:len(short_str)]:
                return long_str
        return None

def get_json(path, unknown_args=[]):
    # Recodring Mechanism.
    with open(path) as fid:
        conf = json.load(fid)
    def get_dict_value(in_dict, key_list):
        if len(key_list) == 0:
            return in_dict
        elif len(key_list) == 1:
            return in_dict[key_list[0]]
        else:
            return get_dict_value(in_dict[key_list[0]], key_list[1:])
    def str_idx(in_dict):
        for key, item in in_dict.items():
            if isinstance(item, str) and len(item)>0 and item[0] == "=":
                in_dict[key] = get_dict_value(conf, item[1:].split("."))
            elif isinstance(item, dict):
                in_dict[key] = str_idx(item)
        return in_dict
    for arg_part, var_part in zip(unknown_args[::2], unknown_args[1::2]): #::2~even integer; 1::2~odd integer
        arg_list = arg_part[2:].split(".")
        print(arg_list)
        leaf_dict = get_dict_value(conf, arg_list[:-1])
        if isinstance(leaf_dict[arg_list[-1]], str):
            leaf_dict[arg_list[-1]] = var_part
        else:
            leaf_dict[arg_list[-1]] = eval(var_part)
    replace_keys = str_idx(conf)
    return conf

def accuracy(output, target, topk=(1,)):
    # Computing the accuracy for the top (TOPK).    
    maxk = max(topk)
    batch_size = target.size(0)
    try:
        _, pred = output.module.topk(maxk, 1, True, True) # The maximum index.
    except:
        _, pred = output.topk(maxk, 1, True, True) # The maximum index.
    pred = pred.t()
    # Squeeze one_hot vector
    if target.ndimension() > 1:
        target = target.max(1)[1]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

class AverageMeter():
    # Recodring Class
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def no_grad(weights_dic):
    for weight in weights_dic:
        weight.requires_grad = False 

def set_grad(weights_dic):
    for weight in weights_dic:
        weight.requires_grad = True

class LRScheduler(object):
    def __init__(self, optimizer, niters, config):
        super(LRScheduler, self).__init__()
        self.mode = config['lr_mode']
        self.warmup_mode = config['warmup_mode'] if 'warmup_mode' in config.keys() else 'linear'
        assert(self.mode in ['step', 'poly', 'cosine'])
        assert(self.warmup_mode in ['linear', 'constant'])
        self.optimizer = optimizer
        self.base_lr = config['base_lr'] if 'base_lr' in config.keys() else 0.1
        self.learning_rate = self.base_lr
        self.niters = niters
        self.step = [int(i) for i in config['step'].split(',')] if 'step' in config.keys() else [30, 60, 90]
        self.decay_factor = config['decay_factor'] if 'decay_factor' in config.keys() else 0.1
        self.targetlr = config['targetlr'] if 'targetlr' in config.keys() else 0.0
        self.power = config['power'] if 'power' in config.keys() else 2.0
        self.warmup_lr = config['warmup_lr'] if 'warmup_lr' in config.keys() else 0.0
        self.max_iter = config['epochs'] * niters
        self.warmup_iters = (config['warmup_epochs'] if 'warmup_epochs' in config.keys() else 0) * niters
    def update(self, i, epoch):
        T = epoch * self.niters + i
        #print(T)
        #print(self.warmup_iters)
        assert (T >= 0 and T <= self.max_iter)
        if self.warmup_iters > T:
            # Warm-up Stage
            if self.warmup_mode == 'linear':
                #print('------------------Here------------------------')
                #print(self.learning_rate, '----', self.warmup_lr, '----', self.base_lr, '-----', self.warmup_iters)
                '''In the warm up, the learning rate growing from 0 '''
                self.learning_rate = self.warmup_lr + (self.base_lr - self.warmup_lr) * T / self.warmup_iters
            elif self.warmup_mode == 'constant':
                self.learning_rate = self.warmup_lr
            else:
                raise NotImplementedError
        else:
            '''In the second phrase, the learning rate decrease from max learning rate.'''
            if self.mode == 'step':
                count = sum([1 for s in self.step if s <= epoch])
                self.learning_rate = self.base_lr * pow(self.decay_factor, count)
            elif self.mode == 'poly':
                self.learning_rate = self.targetlr + (self.base_lr - self.targetlr) * \
                    pow(1 - (T - self.warmup_iters) / (self.max_iter - self.warmup_iters), self.power)
            elif self.mode == 'cosine':
                self.learning_rate = self.targetlr + (self.base_lr - self.targetlr) * \
                    (1 + cos(pi * (T - self.warmup_iters) / (self.max_iter - self.warmup_iters))) / 2
            else:
                raise NotImplementedError
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.learning_rate

class LRScheduler_Doal(object):
    def __init__(self, niters, config):
        super(LRScheduler_Doal, self).__init__()
        self.mode = config['lr_mode']
        self.warmup_mode = config['warmup_mode'] if 'warmup_mode' in config.keys() else 'linear'
        assert(self.mode in ['step', 'poly', 'cosine'])
        assert(self.warmup_mode in ['linear', 'constant'])
        self.base_lr = config['base_lr'] if 'base_lr' in config.keys() else 0.1
        self.learning_rate = self.base_lr
        self.niters = niters
        self.step = [int(i) for i in config['step'].split(',')] if 'step' in config.keys() else [30, 60, 90]
        self.decay_factor = config['decay_factor'] if 'decay_factor' in config.keys() else 0.1
        self.targetlr = config['targetlr'] if 'targetlr' in config.keys() else 0.0
        self.power = config['power'] if 'power' in config.keys() else 2.0
        self.warmup_lr = config['warmup_lr'] if 'warmup_lr' in config.keys() else 0.0
        self.max_iter = config['epochs'] * niters
        self.warmup_iters = (config['warmup_epochs'] if 'warmup_epochs' in config.keys() else 0) * niters
    def update(self, i, epoch):
        T = epoch * self.niters + i
        #print(T)
        #print(self.warmup_iters)
        assert (T >= 0 and T <= self.max_iter)
        if self.warmup_iters > T:
            # Warm-up Stage
            if self.warmup_mode == 'linear':
                #print('------------------Here------------------------')
                #print(self.learning_rate, '----', self.warmup_lr, '----', self.base_lr, '-----', self.warmup_iters)
                '''In the warm up, the learning rate growing from 0 '''
                self.learning_rate = self.warmup_lr + (self.base_lr - self.warmup_lr) * T / self.warmup_iters
            elif self.warmup_mode == 'constant':
                self.learning_rate = self.warmup_lr
            else:
                raise NotImplementedError
        else:
            '''In the second phrase, the learning rate decrease from max learning rate.'''
            if self.mode == 'step':
                count = sum([1 for s in self.step if s <= epoch])
                self.learning_rate = self.base_lr * pow(self.decay_factor, count)
            elif self.mode == 'poly':
                self.learning_rate = self.targetlr + (self.base_lr - self.targetlr) * \
                    pow(1 - (T - self.warmup_iters) / (self.max_iter - self.warmup_iters), self.power)
            elif self.mode == 'cosine':
                self.learning_rate = self.targetlr + (self.base_lr - self.targetlr) * \
                    (1 + cos(pi * (T - self.warmup_iters) / (self.max_iter - self.warmup_iters))) / 2
            else:
                raise NotImplementedError
        return self.learning_rate

def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output

def _is_long(x):
    if hasattr(x, 'data'):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)

def cross_entropy(inputs, target, weight=None, ignore_index=-100, reduction='mean',
                  smooth_eps=None, smooth_dist=None, from_logits=True):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0
    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target) and smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
        else:
            return F.nll_loss(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
    if from_logits:
        # log-softmax of inputs
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = inputs
    masked_indices = None
    num_classes = inputs.size(-1)
    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)
    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)
    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)
    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1. - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)
    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())
    return loss

class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""
    def __init__(self, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, smooth_dist=None, from_logits=True):
        super(CrossEntropyLoss, self).__init__(weight=weight,
                                               ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits
    def forward(self, input, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        return cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index,
                             reduction=self.reduction, smooth_eps=self.smooth_eps,
                             smooth_dist=smooth_dist, from_logits=self.from_logits)

'''
class LASSO:
    def __init__(self, A):
        m, n = A.shape
        self.A = A
        self.m = m
        self.n = n
    def construct_prob(self, b):
        gamma = cvx.Parameter(nonneg=True, value=1e-5)
        x = cvx.Variable(self.n)
        error = cvx.sum_squares(self.A * x - b)
        obj = cvx.Minimize(0.5 * error + gamma * cvx.norm(x, 1))
        prob = cvx.Problem(obj)
        return prob, x
    def solve(self, b):
        prob, x = self.construct_prob(b)
        #prob.solve(solver=cvx.MOSEK)
        prob.solve()
        x_res = np.array(x.value).reshape(-1)
        return x_res
'''

def SK_LASSO(A, b):
    clf = linear_model.Lasso(alpha=1e-5)
    clf.fit(A, b)
    return clf.coef_

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).to(x.device)
        x.div_(keep_prob)
        x.mul_(mask)
    return x

def parameter_name_replace(runmanager):
    # In this document, we transfer from the search space to retraining space.
    state_dict_v2 = copy.deepcopy(runmanager.super_net.net.state_dict())
    for key in runmanager.super_net.net.state_dict().keys():
        num = key.find('_ops.')
        if num != -1:
            num = key.find('_ops.', num+1)
            state_dict_v2[key[:num]+key[num+7:]] = state_dict_v2.pop(key)
            key = key[:num]+key[num+7:]
        try:
            integer = int(key[num]) 
            state_dict_v2[key[:num]+key[num+2:]] = state_dict_v2.pop(key)
        except:
            state_dict_v2 = state_dict_v2
    return state_dict_v2

class CrossEntropyLabelSmooth(nn.Module):
  def __init__(self, num_classes, epsilon, reduction=True):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.reduction = reduction
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)
  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    if self.reduction:
        loss = (-targets * log_probs).mean(0).sum()
    elif not self.reduction:
        loss = (-targets * log_probs).sum(1)
    return loss

def written(acc, epoch, address):
    address_final = os.path.join(address,'record.txt')
    with open(address_final,"a") as f:
        f.write(str(epoch)+'_'+str(acc)+'\n')