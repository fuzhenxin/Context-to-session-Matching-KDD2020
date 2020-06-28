import sys
import os
import time

import pickle as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import utils.evaluation as eva


def train(conf, _model):
    
    if conf['rand_seed'] is not None:
        np.random.seed(conf['rand_seed'])

    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # load data
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'starting loading data')
    train_data_cc, val_data_cc, test_data_cc, test_human_cc = pickle.load(open(conf["data_path"]+"cc.pkl", 'rb'))

    if conf["train_type"]=="cr":
        train_data_cr, val_data_cr, test_data_cr, test_human_cr = pickle.load(open(conf["data_path"]+"cr.pkl", 'rb'))

    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'finish loading data')
    val_batches_cc = reader.build_batches(val_data_cc, conf)
    if conf["train_type"]=="cr":
        val_batches_cr = reader.build_batches(val_data_cr, conf)
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "finish building test batches")

    # refine conf
    batch_num_cc = int(len(train_data_cc['y']) / conf["batch_size"])
    if conf["train_type"]=="cr":
        batch_num_cr = int(len(train_data_cr['y']) / conf["batch_size"])
    val_batch_num_cc = len(val_batches_cc["response"])
    if conf["train_type"]=="cr":
        val_batch_num_cr = len(val_batches_cr["response"])


    print('configurations: %s' %conf)
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'model sucess')

    _graph = _model.build_graph()
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'build graph sucess')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(graph=_graph, config=config) as sess:
        _model.init.run()
        if conf["init_model"]:
            _model.saver_load.restore(sess, conf["init_model"])
            print("sucess init %s" %conf["init_model"])

        average_loss = 0.0
        batch_index = 0
        step = 0
        best_result = (0, 0, 0, 0)

        train_type = conf["train_type"]
        if train_type=="cc":
            g_updates = _model.g_updates_cc
            loss = _model.loss_cc
            global_step = _model.global_step_cc
            learning_rate = _model.learning_rate_cc
            logits = _model.logits_cc

            train_data = train_data_cc
            val_batches = val_batches_cc
            batch_num = batch_num_cc
            val_batch_num = val_batch_num_cc

        elif train_type=="cr":
            g_updates = _model.g_updates_cr
            loss = _model.loss_cr
            global_step = _model.global_step_cr
            learning_rate = _model.learning_rate_cr
            logits = _model.logits_cr

            train_data = train_data_cr
            val_batches = val_batches_cc
            batch_num = batch_num_cr
            val_batch_num = val_batch_num_cc

        elif train_type=="ccr":
            g_updates = _model.g_updates_ccr
            loss = _model.loss_ccr
            global_step = _model.global_step_ccr
            learning_rate = _model.learning_rate_ccr
            logits = _model.logits_ccr

            train_data = train_data_cc
            val_batches = val_batches_cc
            batch_num = batch_num_cc
            val_batch_num = val_batch_num_cc

        else:
            assert False


        for step_i in range(conf["num_scan_data"]):
            #for batch_index in rng.permutation(range(batch_num)):
            print('starting shuffle train data')
            shuffle_train = reader.unison_shuffle(train_data)
            train_batches = reader.build_batches(shuffle_train, conf)
            print('finish building train data')
            for batch_index in range(batch_num):
                feed = {
                    _model.turns1: train_batches["turns1"][batch_index],
                    _model.turns2: train_batches["turns2"][batch_index], 
                    _model.tt_turns_len1: train_batches["tt_turns_len1"][batch_index],
                    _model.every_turn_len1: train_batches["every_turn_len1"][batch_index],
                    _model.tt_turns_len2: train_batches["tt_turns_len2"][batch_index],
                    _model.every_turn_len2: train_batches["every_turn_len2"][batch_index],
                    _model.response: train_batches["response"][batch_index], 
                    _model.response_len: train_batches["response_len"][batch_index],
                    _model.label: train_batches["label"][batch_index],
                    _model.keep_rate: 1.0,
                }

                _, curr_loss = sess.run([g_updates, loss], feed_dict = feed)
                average_loss += curr_loss
                step += 1

                if step % conf["print_step"] == 0 and step > 0:
                    g_step, lr = sess.run([global_step, learning_rate])
                    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'epoch: %d, step: %.5d, lr: %-.6f, loss: %s' %(step_i, g_step, lr, average_loss / conf["print_step"]) )
                    average_loss = 0


            #--------------------------evaluation---------------------------------
            score_file_path = conf['save_path'] + '/score.' + str(step_i)
            score_file = open(score_file_path, 'w')

            for batch_index in range(val_batch_num):
                feed = {
                    _model.turns1: val_batches["turns1"][batch_index],
                    _model.turns2: val_batches["turns2"][batch_index], 
                    _model.tt_turns_len1: val_batches["tt_turns_len1"][batch_index],
                    _model.every_turn_len1: val_batches["every_turn_len1"][batch_index],
                    _model.tt_turns_len2: val_batches["tt_turns_len2"][batch_index],
                    _model.every_turn_len2: val_batches["every_turn_len2"][batch_index],
                    _model.response: val_batches["response"][batch_index], 
                    _model.response_len: val_batches["response_len"][batch_index],
                    _model.keep_rate: 1.0,
                }

                scores = sess.run(logits, feed_dict = feed)
                att_scores = 0.0
                    
                for i in range(conf["batch_size"]):
                    score_file.write(
                        str(scores[i]) + '\t' + 
                        str(val_batches["label"][batch_index][i]) + '\n')
            score_file.close()

            result = eva.evaluate(score_file_path)
            print(time.strftime('%Y-%m-%d %H:%M:%S result: ',time.localtime(time.time())), *result)


            if result[1] + result[2] > best_result[1] + best_result[2]:
                best_result = result
                _save_path = _model.saver_save.save(sess, conf["save_path"] + "/model", global_step=step_i)
                print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "succ saving model in " + _save_path)
            print(time.strftime('%Y-%m-%d %H:%M:%S best result',time.localtime(time.time())), *best_result)
                    
                

