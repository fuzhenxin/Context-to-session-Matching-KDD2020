
import tensorflow as tf
import models.layer_utils as layer_utils
import models.match_utils as match_utils

class Options():
    def __init__(self):
        self.suffix = "quora"
        self.fix_word_vec= True
        self.isLower= True
        self.max_sent_length = 50
        self.max_char_per_word = 10

        self.with_char= True
        self.char_emb_dim = 20
        self.char_lstm_dim = 40


        self.batch_size = 60
        self.max_epochs = 20
        self.dropout_rate = 0.1
        self.learning_rate = 0.0005
        self.optimize_type = "adam"
        self.lambda_l2 = 0.0
        self.grad_clipper = 10.0

        self.context_layer_num = 1
        self.context_lstm_dim = 100
        self.aggregation_layer_num = 1
        self.aggregation_lstm_dim = 100

        self.with_full_match= True
        self.with_maxpool_match = False
        self.with_max_attentive_match = False
        self.with_attentive_match= True

        self.with_cosine= True
        self.with_mp_cosine= True
        self.cosine_MP_dim = 5

        self.att_dim = 50
        self.att_type = "symmetric"

        self.highway_layer_num = 1
        self.with_highway= True
        self.with_match_highway= True
        self.with_aggregation_highway= True

        self.use_cudnn= True

        self.with_moving_average = False



def bimpm_model(input_x, input_x_mask, input_y, input_y_mask, word_emb, keep_rate):

        input_dim = 200
        dropout_rate = 0.0
        with_highway = True
        highway_layer_num = 1
        passage_lengths = None
        question_lengths = None
        is_training = False

        options  = Options()

        in_question_repres = tf.nn.embedding_lookup(word_emb, input_x) # [batch_size, question_len, word_dim]
        in_passage_repres = tf.nn.embedding_lookup(word_emb, input_y) # [batch_size, passage_len, word_dim]

        input_x_mask = tf.cast(input_x_mask, tf.float32)
        input_y_mask = tf.cast(input_y_mask, tf.float32)

        input_shape = tf.shape(input_x)
        batch_size = input_shape[0]
        question_len = input_shape[1]
        input_shape = tf.shape(input_y)
        passage_len = input_shape[1]

        passage_lengths = tf.constant([100, ], shape=(1,), dtype=tf.int32)
        passage_lengths = tf.tile(passage_lengths, [batch_size])
        question_lengths = passage_lengths
            

        if is_training:
            in_question_repres = tf.nn.dropout(in_question_repres, (1 - dropout_rate))
            in_passage_repres = tf.nn.dropout(in_passage_repres, (1 - dropout_rate))

        mask = input_y_mask
        question_mask = input_x_mask

        # ======Highway layer======
        if with_highway:
            with tf.variable_scope("input_highway"):
                in_question_repres = match_utils.multi_highway_layer(in_question_repres, input_dim, highway_layer_num)
                tf.get_variable_scope().reuse_variables()
                in_passage_repres = match_utils.multi_highway_layer(in_passage_repres, input_dim, highway_layer_num)

        # in_question_repres = tf.multiply(in_question_repres, tf.expand_dims(question_mask, axis=-1))
        # in_passage_repres = tf.multiply(in_passage_repres, tf.expand_dims(mask, axis=-1))

        # ========Bilateral Matching=====
        (match_representation, match_dim) = match_utils.bilateral_match_func(in_question_repres, in_passage_repres,
                        question_lengths, passage_lengths, question_mask, mask, input_dim, is_training, options=options)

        #========Prediction Layer=========
        # match_dim = 4 * self.options.aggregation_lstm_dim
        w_0 = tf.get_variable("w_0", [match_dim, match_dim/2], dtype=tf.float32)
        b_0 = tf.get_variable("b_0", [match_dim/2], dtype=tf.float32)


        # if is_training: match_representation = tf.nn.dropout(match_representation, (1 - options.dropout_rate))
        logits = tf.matmul(match_representation, w_0) + b_0
        return logits

