import tensorflow as tf
import numpy as np
import pickle as pickle

import utils.layers as layers
import utils.operations as op


def cc_model(input_x, input_x_mask, input_x_len, input_x2, input_x_mask2, input_x_len2, word_emb, conf, con_c):


    #a list of length max_turn_num, every element is a tensor with shape [batch, max_turn_len]
    list_turn_t1 = tf.unstack(input_x, axis=1) 
    list_turn_length1 = tf.unstack(input_x_len, axis=1)
    list_turn_length1 = [tf.sequence_mask(i, conf["max_turn_len"]) for i in list_turn_length1]
    list_turn_length1 = [tf.cast(i, tf.float32) for i in list_turn_length1]

    list_turn_t2 = tf.unstack(input_x2, axis=1) 
    list_turn_length2 = tf.unstack(input_x_len2, axis=1)
    list_turn_length2 = [tf.sequence_mask(i, conf["max_turn_len"]) for i in list_turn_length2]
    list_turn_length2 = [tf.cast(i, tf.float32) for i in list_turn_length2]

    if con_c:
        list_turn_t1 = tf.reshape(input_x, [conf["batch_size"], conf["max_turn_num"]*conf["max_turn_len"]])
        list_turn_t1 = [list_turn_t1]
        list_turn_t2 = tf.reshape(input_x2, [conf["batch_size"], conf["max_turn_num"]*conf["max_turn_len"]])
        list_turn_t2 = [list_turn_t2]
        list_turn_length1 = tf.cast(tf.sequence_mask(input_x_len, conf["max_turn_len"]), tf.float32)
        list_turn_length1 = tf.reshape(list_turn_length1, [conf["batch_size"], conf["max_turn_num"]*conf["max_turn_len"]])
        list_turn_length1 = [list_turn_length1]
        list_turn_length2 = tf.cast(tf.sequence_mask(input_x_len2, conf["max_turn_len"]), tf.float32)
        list_turn_length2 = tf.reshape(list_turn_length2, [conf["batch_size"], conf["max_turn_num"]*conf["max_turn_len"]])
        list_turn_length2 = [list_turn_length2]



    #for every turn_t calculate matching vector
    trans_u1, trans_u2 = [], []
    for turn_t, t_turn_length in zip(list_turn_t1, list_turn_length1):
        Hu = tf.nn.embedding_lookup(word_emb, turn_t) #[batch, max_turn_len, emb_size]
        #Hu = turn_t
        if conf['is_positional'] and conf['stack_num'] > 0:
            with tf.variable_scope('positional_', reuse=tf.AUTO_REUSE):
                Hu = op.positional_encoding_vector(Hu, max_timescale=10)
        for index in range(conf['stack_num']):
            with tf.variable_scope('self_stack_cc' + str(index), reuse=tf.AUTO_REUSE):
                Hu = layers.block(
                    Hu, Hu, Hu,
                    Q_lengths=t_turn_length, K_lengths=t_turn_length, input_mask=True)
        trans_u1.append(Hu)

    for turn_r, r_turn_length in zip(list_turn_t2, list_turn_length2):
        Hu = tf.nn.embedding_lookup(word_emb, turn_r) #[batch, max_turn_len, emb_size]
        #Hu = turn_r
        if conf['is_positional'] and conf['stack_num'] > 0:
            with tf.variable_scope('positional_', reuse=tf.AUTO_REUSE):
                Hu = op.positional_encoding_vector(Hu, max_timescale=10)
        for index in range(conf['stack_num']):
            with tf.variable_scope('self_stack_cc' + str(index), reuse=tf.AUTO_REUSE):
                Hu = layers.block(
                    Hu, Hu, Hu,
                    Q_lengths=r_turn_length, K_lengths=r_turn_length, input_mask=True)
        trans_u2.append(Hu)

    final_info_all = []
    sim_turns_all = []
    for t_inedx, (turn_t, t_turn_length, Hu) in enumerate(zip(list_turn_t1, list_turn_length1, trans_u1)):
        sim_turns = []
        for r_index, (turn_r, r_turn_length, Hr) in enumerate(zip(list_turn_t2, list_turn_length2, trans_u2)):

            with tf.variable_scope('u_attentd_r_' + str(index)):
                try:
                    u_a_r = layers.block(
                        Hu, Hr, Hr,
                        Q_lengths=t_turn_length, K_lengths=r_turn_length, input_mask=True)
                except ValueError:
                    tf.get_variable_scope().reuse_variables()
                    u_a_r = layers.block(
                        Hu, Hr, Hr,
                        Q_lengths=t_turn_length, K_lengths=r_turn_length, input_mask=True)
                    

            with tf.variable_scope('r_attend_u_' + str(index)):
                try:
                    r_a_u = layers.block(
                        Hr, Hu, Hu,
                        Q_lengths=r_turn_length, K_lengths=t_turn_length, input_mask=True)
                except ValueError:
                    tf.get_variable_scope().reuse_variables()
                    r_a_u = layers.block(
                        Hr, Hu, Hu,
                        Q_lengths=r_turn_length, K_lengths=t_turn_length, input_mask=True)
        
            # u_a_r batch_size turn emb
            u_a_r = tf.stack([u_a_r, Hu], axis=-1)
            r_a_u = tf.stack([r_a_u, Hr], axis=-1)
        
            #calculate similarity matrix
            with tf.variable_scope('similarity', reuse=tf.AUTO_REUSE):
                #sim shape [batch, max_turn_len, max_turn_len, 2*stack_num+1]
                #sim shape [batch, max_turn_len, max_turn_len, 2]
                sim = tf.einsum('biks,bjks->bijs', r_a_u, u_a_r) / tf.sqrt(200.0)
                sim = layers.CNN_FZX(sim)
            final_info_all.append(sim)

    att_weight_print = None
    if not con_c:
        # final_info_all
        final_info_all = tf.stack(final_info_all, axis=1)  # 100 9 144
        max_nei = 5
        gcn_size = conf["max_turn_num"]*conf["max_turn_num"]
        turn_size = conf["max_turn_num"]
        m1 = [ [] for i in range(gcn_size)]
        m_pos = [ [] for i in range(gcn_size)]
        m1_len = [ 0 for i in range(gcn_size)]
        for i in range(turn_size):
            for j in range(turn_size):
                cur_index = i*turn_size+j
                m1[cur_index].append(cur_index)
                m_pos[cur_index].extend([i,j])
                if cur_index%turn_size!=0:
                    m1[cur_index].append(cur_index-1)
                    m_pos[cur_index].extend([i-1,j])
                if cur_index%turn_size!=turn_size-1:
                    m1[cur_index].append(cur_index+1)
                    m_pos[cur_index].extend([i+1,j])
                if i!=0:
                    m1[cur_index].append(cur_index-turn_size)
                    m_pos[cur_index].extend([i,j-1])
                if i!=turn_size-1:
                    m1[cur_index].append(cur_index+turn_size)
                    m_pos[cur_index].extend([i,j+1])
                m1_len[cur_index] = len(m1[cur_index])
                if m1_len[cur_index]<max_nei:
                    m1[cur_index].extend([cur_index for k in range(max_nei-m1_len[cur_index])])
                    for k in range(max_nei-m1_len[cur_index]): m_pos[cur_index].extend([i,j])
        # m1 25 5
        # m1_len 25

        m1 = tf.constant(m1, dtype=tf.int32) # 25 5
        m1_len = tf.constant(m1_len, dtype=tf.int32)
        m_pos = tf.constant(m_pos, dtype=tf.int32)

        def gan(input_m, adjm, adjm_len, adjm_pos, gcn_size, turn_size, max_nei):
            #return input_m
            batch_size_gnn = tf.shape(input_m)[0]
            mask_value = tf.cast(tf.sequence_mask(adjm_len, max_nei), tf.float32) # 25 5
            res_all = []
            for gan_index in range(4):
                with tf.variable_scope('gan_layer'+str(gan_index), reuse=tf.AUTO_REUSE):
                    role_emb1 = tf.get_variable(name="gnn_role_emb1", shape=[turn_size, conf["role_dim"]], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=1))
                    role_emb2 = tf.get_variable(name="gnn_role_emb2", shape=[turn_size, conf["role_dim"]], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=1))

                    input_m_exp = tf.expand_dims(input_m, axis=2) # bs 25 1 144
                    input_m_exp = tf.tile(input_m_exp, [1, 1, max_nei, 1]) # bs 25 5 144

                    nei_rep = tf.gather(input_m, adjm, axis=1) # bs 25*5 144
                    nei_rep = tf.reshape(nei_rep, [tf.shape(input_m)[0], gcn_size, max_nei, -1]) # bs 25 5 144

                    att1 = tf.layers.dense(nei_rep, 128, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="gcn") # bs 25 5 128
                    att2 = tf.layers.dense(input_m_exp, 128, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="gcn") # bs 25 5 128


                    pos_index11 = tf.gather(adjm_pos, [0,], axis=1)
                    pos_index12 = tf.gather(adjm_pos, [1,], axis=1)
                    pos_index11 = tf.tile(pos_index11, [1, max_nei])
                    pos_index12 = tf.tile(pos_index12, [1, max_nei])

                    pos_index21 = tf.gather(adjm_pos, [0,2,4,6,8], axis=1)
                    pos_index22 = tf.gather(adjm_pos, [1,3,5,7,9], axis=1)

                    pos_index11 = tf.gather(role_emb1, pos_index11) # 25 5 30
                    pos_index12 = tf.gather(role_emb2, pos_index12) # 25 5 30
                    pos_index21 = tf.gather(role_emb1, pos_index21) # 25 5 30
                    pos_index22 = tf.gather(role_emb2, pos_index22) # 25 5 30

                    pos_index11 = tf.tile(tf.expand_dims(pos_index11, axis=0), [batch_size_gnn,1,1,1])
                    pos_index12 = tf.tile(tf.expand_dims(pos_index12, axis=0), [batch_size_gnn,1,1,1])
                    pos_index21 = tf.tile(tf.expand_dims(pos_index21, axis=0), [batch_size_gnn,1,1,1])
                    pos_index22 = tf.tile(tf.expand_dims(pos_index22, axis=0), [batch_size_gnn,1,1,1])


                    att = tf.concat([att1, att2], axis=-1)
                    #att = tf.concat([att1, att2, pos_index11, pos_index12, pos_index21, pos_index22], axis=-1)
                    att = tf.layers.dense(att, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="gcna") # bs 25 5 128
                    att = tf.reshape(att, [-1, gcn_size, max_nei])
                    att = tf.nn.leaky_relu(att) # bs 25 5

                    att = att * tf.expand_dims(mask_value, axis=0)
                    att = tf.nn.softmax(att, axis=2) # bs 25 5
                    att = att * tf.expand_dims(mask_value, axis=0)

                    nei_rep2 = tf.layers.dense(nei_rep, 128, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="gcnl") # bs 25 5 128
                    nei_rep11 = tf.layers.dense(input_m, 128, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="gcnl") # bs 25 5 128
                    nei_rep2 = nei_rep2 * tf.expand_dims(tf.expand_dims(mask_value, axis=0), axis=-1)

                    res = tf.einsum('bdik,bdi->bdk', nei_rep2, att) # bs 25 128

                    att_input = res+nei_rep11
                    att_out = tf.layers.dense(att_input, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="att"+str(i))
                    att_out = tf.nn.sigmoid(att_out)
                    print_weight = att_out
                    # att_out not used

                    res = res + nei_rep11
                    input_m = res
                    res_all.append(res)
            res_all = tf.concat(res_all, axis=-1)

            return res_all, print_weight

        gan_res, att_weight_print = gan(final_info_all, m1, m1_len, m_pos, gcn_size, turn_size, max_nei)


        final_info_all = gan_res
        final_info_role = []

        role_emb1 = tf.get_variable(name="role_emb1", shape=[len(list_turn_t1), conf["role_dim"]], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=1))
        role_emb2 = tf.get_variable(name="role_emb2", shape=[len(list_turn_t2), conf["role_dim"]], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=1))
        for i, ii in enumerate(list_turn_t1):
            for j, jj in enumerate(list_turn_t2):
                role_con = tf.concat([role_emb1[i], role_emb2[j]], axis=0)
                final_info_role.append(role_con)
        final_info_role = tf.stack(final_info_role, axis=0) # 9 50
        final_info_role = tf.expand_dims(final_info_role, 0) # 1 9 50
        final_info_role = tf.tile(final_info_role, [tf.shape(final_info_all)[0], 1, 1], name="role_con")
        final_info_all_att = tf.concat([final_info_role, final_info_all], axis=2)

        final_info_all_att = tf.reshape(final_info_all_att, [-1, final_info_all_att.get_shape()[-1]]) # bs*9 144
        final_info_all_att = tf.layers.dense(final_info_all_att, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())
        final_info_all_att = tf.squeeze(final_info_all_att, [1])
        final_info_all_att = tf.reshape(final_info_all_att, [-1, final_info_all.get_shape()[1]]) # 100 9
        final_info_all_att = tf.nn.softmax(final_info_all_att, axis=1)

        final_info_all_att = tf.expand_dims(final_info_all_att, -1)
        final_info_all_max = tf.reduce_max(final_info_all, axis=1)
        final_info_all_mean = tf.reduce_mean(final_info_all, axis=1)
        final_info_all =  final_info_all * final_info_all_att
        final_info_all = tf.reduce_sum(final_info_all, axis=1)

        final_info_all = tf.concat([final_info_all_mean, final_info_all_max, final_info_all], axis=1)

    else:
        final_info_all = final_info_all[0]
    return final_info_all, att_weight_print
