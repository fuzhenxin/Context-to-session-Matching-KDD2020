import sys
import os
import time

import pickle as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
#import models.last_net as net
import models.net as net
import utils.evaluation as eva
#for douban
#import utils.douban_evaluation as eva

import bin.train_and_evaluate as train
import bin.test_and_evaluate as test

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# configure

data_pre = "data_ali"
output_pre = "../output_ali"
vocab_size = 36105

if sys.argv[1]=="train":
    init_model = None
    test_mod = "cc"
else:
    init_model = sys.argv[3]
    test_mod = sys.argv[2]

conf = {
    "data_path": data_pre + "/data.cc.",
    "save_path": output_pre + "/5.19.cr.ioi.55",
    "word_emb_init": data_pre + "/glove.cc.cr.pkl",
    "init_model": init_model , #"output/model.ckpt", #should be set for test

    "train_type": "cr",
    "cr_model": "IOI",
    "cc_model": "cc",
    "fusion": "fusion",

    "con": "con2",
    "con2_att": "gat_mmcr",

    "rand_seed": None,
    "print_step": 100,

    "drop_dense": None,
    "drop_attention": None,

    "is_mask": True,
    "is_layer_norm": True,
    "is_positional": True,

    "stack_num": 3,
    "stack_num_cr": 3,
    "stack_num_h": 0,
    "attention_type": "dot",

    "learning_rate": 1e-4,
    "vocab_size": vocab_size,
    "emb_size": 200,
    "batch_size": 100,

    "max_turn_num": 5,
    "max_turn_len": 20,

    "max_to_keep": 1,
    "num_scan_data": 10,
    "_EOS_": 1,
    "final_n_class": 1,

    "test_mod": test_mod,
    "role_dim": 25,
}

if sys.argv[1]=="train":
    model = net.Net(conf, is_train=True)
    train.train(conf, model)
else:
    #test and evaluation, init_model in conf should be set
    model = net.Net(conf, is_train=False)
    test.test(conf, model)
