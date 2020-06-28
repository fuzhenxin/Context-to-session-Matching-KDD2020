import tensorflow as tf
import numpy as np
import pickle as pickle

import utils.layers as layers
import utils.operations as op

from models.ioi_net import ioi_model
from models.dam_net import dam_model
from models.smn_net import smn_model
from models.cc_net import cc_model
from models.bimpm_net import bimpm_model



class Net(object):
    '''Add positional encoding(initializer lambda is 0),
       cross-attention, cnn integrated and grad clip by value.

    Attributes:
        conf: a configuration paramaters dict
        word_embedding_init: a 2-d array with shape [vocab_size+1, emb_size]
    '''
    def __init__(self, conf, is_train=False):
        self._graph = tf.Graph()
        self._conf = conf
        # con2 con3 con4 gru
        self.final_ag_att = conf["con2_att"]
        self.is_train = is_train
        self.cr_model = conf["cr_model"]
        self.cc_model = conf["cc_model"]
        self.con_c = False

        if self._conf['word_emb_init'] is not None:
            print('loading word emb init')
            self._word_embedding_init = pickle.load(open(self._conf['word_emb_init'], 'rb'))
        else:
            self._word_embedding_init = None

    def build_graph(self):
        with self._graph.as_default():
            rand_seed = self._conf['rand_seed']
            tf.set_random_seed(rand_seed)

            #word embedding
            if self._word_embedding_init is not None:
                word_embedding_initializer = tf.constant_initializer(self._word_embedding_init)
            else:
                word_embedding_initializer = tf.random_normal_initializer(stddev=0.1)

            self._word_embedding = tf.get_variable(
                name='word_embedding',
                shape=[self._conf['vocab_size']+1, self._conf['emb_size']],
                dtype=tf.float32,
                initializer=word_embedding_initializer)

            #define placehloders
            self.turns1 = tf.placeholder(tf.int32, shape=[self._conf["batch_size"], self._conf["max_turn_num"], self._conf["max_turn_len"]], name="turns1")
            self.tt_turns_len1 = tf.placeholder(tf.int32, shape=[self._conf["batch_size"]], name="tt_turns_len1")
            self.every_turn_len1 = tf.placeholder(tf.int32, shape=[self._conf["batch_size"], self._conf["max_turn_num"]], name="every_turn_len1")
            self.turns2 = tf.placeholder(tf.int32, shape=[self._conf["batch_size"], self._conf["max_turn_num"], self._conf["max_turn_len"]], name="turns2")
            self.tt_turns_len2 = tf.placeholder(tf.int32, shape=[self._conf["batch_size"]], name="tt_turns_len2")
            self.every_turn_len2 = tf.placeholder(tf.int32, shape=[self._conf["batch_size"], self._conf["max_turn_num"]], name="every_turn_len2")
            self.response = tf.placeholder(tf.int32, shape=[self._conf["batch_size"], self._conf["max_turn_len"]], name="response")
            self.response_len = tf.placeholder(tf.int32, shape=[self._conf["batch_size"]], name="response_len")
            self.keep_rate = tf.placeholder(tf.float32, [], name="keep_rate") 

            self.label = tf.placeholder(tf.float32, shape=[self._conf["batch_size"]])

            self.turns1_e = tf.nn.embedding_lookup(self._word_embedding, self.turns1)
            self.turns2_e = tf.nn.embedding_lookup(self._word_embedding, self.turns2)
            self.response_e = tf.nn.embedding_lookup(self._word_embedding, self.response)


            # SMN
            if self.cr_model=="SMN":
                input_x = self.turns1
                input_y = self.response
                final_info_cr = smn_model(input_x, None, input_y, None, self._word_embedding, self.keep_rate, self._conf, x_len=self.every_turn_len1, y_len=self.response_len)
                with tf.variable_scope('final_smn_mlp_cr'):
                    final_info_cr = tf.layers.dense(final_info_cr, 50, kernel_initializer=tf.contrib.layers.xavier_initializer())           

            # DAM
            elif self.cr_model=="DAM":
                input_x = self.turns1
                input_y = self.response
                final_info_cr = dam_model(input_x, None, input_y, None, self._word_embedding, self.keep_rate, self._conf, x_len=self.every_turn_len1, y_len=self.response_len)
                with tf.variable_scope('final_esim_mlp_cr'):
                    final_info_cr = tf.layers.dense(final_info_cr, 50, kernel_initializer=tf.contrib.layers.xavier_initializer())           

            # IOI
            elif self.cr_model=="IOI":
                input_x = tf.reshape(self.turns1, [self._conf["batch_size"], -1])
                input_x_mask = tf.sequence_mask(self.every_turn_len1, self._conf["max_turn_len"])
                input_x_mask = tf.reshape(input_x_mask, [self._conf["batch_size"], -1])

                input_x2 = tf.reshape(self.turns2, [self._conf["batch_size"], -1])
                input_x_mask2 = tf.sequence_mask(self.every_turn_len2, self._conf["max_turn_len"])
                input_x_mask2 = tf.reshape(input_x_mask2, [self._conf["batch_size"], -1])

                input_y = self.response
                input_y_mask = tf.sequence_mask(self.response_len, self._conf["max_turn_len"])

                final_info_cr, final_info_cr_ioi = ioi_model(input_x, input_x_mask, input_y, input_y_mask, self._word_embedding, self.keep_rate, self._conf)
                with tf.variable_scope('final_esim_mlp_cr'):
                    final_info_cr = tf.layers.dense(final_info_cr, 50, kernel_initializer=tf.contrib.layers.xavier_initializer())



            if self.cc_model=="cc":
                input_x =self.turns1
                input_x_mask = tf.sequence_mask(self.every_turn_len1, self._conf["max_turn_len"])
                #input_x_mask = tf.reshape(input_x_mask, [-1, self._conf["max_turn_num"]*self._conf["max_turn_len"]])
                input_x_len = self.every_turn_len1

                input_x2 = self.turns2
                input_x_mask2 = tf.sequence_mask(self.every_turn_len2, self._conf["max_turn_len"])
                #input_x_mask2 = tf.reshape(input_x_mask2, [-1, self._conf["max_turn_num_s"]*self._conf["max_turn_len"]])
                input_x_len2 = self.every_turn_len2

                final_info_cc, self.att_weight_print = cc_model(input_x, input_x_mask, input_x_len, input_x2, input_x_mask2, input_x_len2, self._word_embedding, self._conf, con_c=self.con_c)
                with tf.variable_scope('final_mlp_cc'):
                    final_info_cc = tf.layers.dense(final_info_cc, 50, kernel_initializer=tf.contrib.layers.xavier_initializer())

            elif self.cc_model=="onesent":
                input_x =self.turns1
                input_x_mask = tf.sequence_mask(self.every_turn_len1, self._conf["max_turn_len"])
                #input_x_mask = tf.reshape(input_x_mask, [-1, self._conf["max_turn_num"]*self._conf["max_turn_len"]])
                input_x_len = self.every_turn_len1

                input_x2 = self.turns2
                input_x_mask2 = tf.sequence_mask(self.every_turn_len2, self._conf["max_turn_len"])
                #input_x_mask2 = tf.reshape(input_x_mask2, [-1, self._conf["max_turn_num_s"]*self._conf["max_turn_len"]])
                input_x_len2 = self.every_turn_len2

                final_info_cc = cc_model(input_x, input_x_mask, input_x_len, input_x2, input_x_mask2, input_x_len2, self._word_embedding, self._conf, con_c=True)
                with tf.variable_scope('final_mlp_onesent_cc'):
                    final_info_cc = tf.layers.dense(final_info_cc, 50, kernel_initializer=tf.contrib.layers.xavier_initializer())

            elif self.cc_model=="bimpm":
                input_x = tf.reshape(self.turns1, [self._conf["batch_size"], -1])
                input_x_mask = tf.sequence_mask(self.every_turn_len1, self._conf["max_turn_len"])
                input_x_mask = tf.reshape(input_x_mask, [self._conf["batch_size"], -1])
                input_y = tf.reshape(self.turns2, [self._conf["batch_size"], -1])
                input_y_mask = tf.sequence_mask(self.every_turn_len2, self._conf["max_turn_len"])
                input_y_mask = tf.reshape(input_y_mask, [self._conf["batch_size"], -1])

                with tf.variable_scope('final_bimpm_cc_cr'):
                    final_info_cc = bimpm_model(input_x, input_x_mask, input_y, input_y_mask, self._word_embedding, self.keep_rate)
                    final_info_cc = tf.layers.dense(final_info_cc, 50, kernel_initializer=tf.contrib.layers.xavier_initializer())


            #loss and train
            with tf.variable_scope('loss_cc'):
                self.loss_cc, self.logits_cc = layers.loss(final_info_cc, self.label)

                self.global_step_cc = tf.Variable(0, trainable=False)
                initial_learning_rate = self._conf['learning_rate']
                self.learning_rate_cc = tf.train.exponential_decay(
                    initial_learning_rate,
                    global_step=self.global_step_cc,
                    decay_steps=5000,
                    decay_rate=0.9,
                    staircase=True)

                Optimizer_cc = tf.train.AdamOptimizer(self.learning_rate_cc)
                self.optimizer_cc = Optimizer_cc.minimize(self.loss_cc)

                #self.all_operations = self._graph.get_operations()
                self.grads_and_vars_cc = Optimizer_cc.compute_gradients(self.loss_cc)

                self.capped_gvs_cc = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in self.grads_and_vars_cc if grad!=None]
                self.g_updates_cc = Optimizer_cc.apply_gradients(
                    self.capped_gvs_cc,
                    global_step=self.global_step_cc)
    
            with tf.variable_scope('loss_cr'):
                if self.cr_model=="IOI":
                    loss_list = []
                    logits_list = []
                    for i,j in enumerate(final_info_cr_ioi):
                        with tf.variable_scope("loss"+str(i)): loss_per, logits_per = layers.loss(j, self.label)
                        loss_list.append(loss_per)
                        logits_list.append(logits_per)
                    self.loss_cr =sum([((idx+1)/7.0)*item for idx, item in enumerate(loss_list)])
                    self.logits_cr = sum(logits_list)
                else:
                    self.loss_cr, self.logits_cr = layers.loss(final_info_cr, self.label)

                self.global_step_cr = tf.Variable(0, trainable=False)
                initial_learning_rate = self._conf['learning_rate']
                self.learning_rate_cr = tf.train.exponential_decay(
                    initial_learning_rate,
                    global_step=self.global_step_cr,
                    decay_steps=10000,
                    decay_rate=0.9,
                    staircase=True)

                Optimizer_cr = tf.train.AdamOptimizer(self.learning_rate_cr)
                self.optimizer_cr = Optimizer_cr.minimize(self.loss_cr)

                self.grads_and_vars_cr = Optimizer_cr.compute_gradients(self.loss_cr)

                self.capped_gvs_cr = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in self.grads_and_vars_cr if grad!=None]
                self.g_updates_cr = Optimizer_cr.apply_gradients(self.capped_gvs_cr, global_step=self.global_step_cr)

            with tf.variable_scope('loss_ccr'):


                if self._conf["fusion"]=="fusion":
                    final_att = tf.concat([final_info_cc, final_info_cr], axis=1)
                    final_att = tf.layers.dense(final_att, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="nosave")
                    final_att = tf.nn.sigmoid(final_att)
                    self.final_att_print = final_att
                    final_info_ccr = final_info_cc*final_att+final_info_cr*(1-final_att)
                elif self._conf["fusion"]=="con":
                    #print(final_info_cr.shape)
                    final_att = tf.concat([final_info_cc, final_info_cr], axis=1)
                    final_att = tf.layers.dense(final_att, final_info_cr.shape[-1], kernel_initializer=tf.contrib.layers.xavier_initializer(), name="nosave")
                    final_att = tf.nn.sigmoid(final_att)
                    self.final_att_print = final_att
                    final_info_ccr = tf.concat([final_info_cr, final_info_cc], axis=1)
                elif self._conf["fusion"]=="none":
                    final_info_ccr = final_info_cc + final_info_cr
                else:
                    assert False

                self.loss_ccr, self.logits_ccr = layers.loss(final_info_ccr, self.label)
                self.loss_ccr += self.loss_cr
                self.logits_ccr += self.logits_cr

                self.global_step_ccr = tf.Variable(0, trainable=False)
                initial_learning_rate = self._conf['learning_rate']
                self.learning_rate_ccr = tf.train.exponential_decay(
                    initial_learning_rate,
                    global_step=self.global_step_ccr,
                    decay_steps=5000,
                    decay_rate=0.9,
                    staircase=True)

                Optimizer_ccr = tf.train.AdamOptimizer(self.learning_rate_ccr)
                self.optimizer_ccr = Optimizer_ccr.minimize(self.loss_ccr)

                self.grads_and_vars_ccr = Optimizer_ccr.compute_gradients(self.loss_ccr)

                self.capped_gvs_ccr = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in self.grads_and_vars_ccr if grad!=None]
                self.g_updates_ccr = Optimizer_ccr.apply_gradients(self.capped_gvs_ccr, global_step=self.global_step_ccr)

            self.all_variables = tf.global_variables()
            self.init = tf.global_variables_initializer()
            self.saver_load = tf.train.Saver(max_to_keep = self._conf["max_to_keep"])
            self.saver_save = self.saver_load
            
            self.all_operations = self._graph.get_operations()

        return self._graph
