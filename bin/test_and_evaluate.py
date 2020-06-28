import sys
import os
import time

import pickle as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import utils.evaluation as eva
import utils.human_evaluation as h_eva


def test(conf, _model):
    
    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # load data
    print('starting loading data')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    train_data, val_data, test_data, test_data_human = pickle.load(open(conf["data_path"]+"cc.pkl", 'rb'))    
    print('finish loading data')

    if conf["test_mod"]=="TestRetrvCand":
        test_data = test_data_human
        score_test = "score.test.human"
    elif conf["test_mod"]=="TestRandNegCand":
        test_data = test_data
        score_test = "score.test"
    else:
        assert False

    test_batches = reader.build_batches(test_data, conf)

    print("finish building test batches")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # refine conf
    test_batch_num = len(test_batches["response"])

    print('configurations: %s' %conf)


    _graph = _model.build_graph()
    print('build graph sucess')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    with tf.Session(graph=_graph) as sess:

        _model.init.run()
        _model.saver_load.restore(sess, conf["init_model"])
        print("sucess init %s" %conf["init_model"])

        test_type = conf["train_type"]
        if test_type=="cc":
            logits = _model.logits_cc
        elif test_type=="cr":
            logits = _model.logits_cr
        elif test_type=="ccr":
            logits = _model.logits_ccr

        score_file_path = conf['save_path'] + '/' + score_test
        score_file = open(score_file_path, 'w')


        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'starting test')
        for batch_index in range(test_batch_num):
            feed = {
                _model.turns1: test_batches["turns1"][batch_index],
                _model.turns2: test_batches["turns2"][batch_index], 
                _model.tt_turns_len1: test_batches["tt_turns_len1"][batch_index],
                _model.every_turn_len1: test_batches["every_turn_len1"][batch_index],
                _model.tt_turns_len2: test_batches["tt_turns_len2"][batch_index],
                _model.every_turn_len2: test_batches["every_turn_len2"][batch_index],
                _model.response: test_batches["response"][batch_index], 
                _model.response_len: test_batches["response_len"][batch_index],
                _model.label: test_batches["label"][batch_index],
                _model.keep_rate: 1.0,
            }
            scores = sess.run(logits, feed_dict = feed)

            for i in range(conf["batch_size"]):
                score_file.write(
                    str(scores[i]) + '\t' + 
                    str(test_batches["label"][batch_index][i]) + '\n')

        score_file.close()
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'finish test')

        #write evaluation result
        print(conf["test_mod"])
        if conf["test_mod"]=="TestRandNegCand":
            result = eva.evaluate(score_file_path)
            print("MRR: {:01.4f} P2@1 {:01.4f} R@1 {:01.4f} r@2 {:01.4f} r@5 {:01.4f}".format(*result))
        elif conf["test_mod"]=="TestRetrvCand":
            data_dir = "/".join(conf["data_path"].split("/")[:-1])
            result = h_eva.evaluate_human(score_file_path, data_dir)
            print("MAP: {:01.4f} MRR: {:01.4f} P@1 {:01.4f} R@1 {:01.4f} r@2 {:01.4f} r@5 {:01.4f}".format(*result))
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'finish evaluation')
