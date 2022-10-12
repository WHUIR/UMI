import os
import tensorflow as tf
from tensorflow.contrib import layers
from utils import hard_attention, dual_softmax_routing, _user_profile_interest_attention


class Model(object):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="DNN"):
        self.model_flag = flag
        self.reg = False
        self.batch_size = batch_size
        self.n_mid = n_mid
        with tf.name_scope('Inputs'):
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.user_gender = tf.placeholder(tf.int32, [None, ], name='user_gender')
            self.user_age = tf.placeholder(tf.int32, [None, ], name='user_age')
            self.user_occup = tf.placeholder(tf.int32, [None, ], name='user_occup')

            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.mid_negs = tf.placeholder(tf.int32, [None, None], name='mid_negs')

            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.lr = tf.placeholder(tf.float64, [])

        self.mask_length = tf.cast(tf.reduce_sum(self.mask, -1), dtype=tf.int32)

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, embedding_dim], trainable=True)
            self.user_gender_embedding_matrix = tf.get_variable("user_gender_embedding_matrix",
                                                                [2, 8],
                                                                trainable=True)
            self.user_age_embedding_matrix = tf.get_variable("user_age_embedding_matrix", [7, 8],
                                                             trainable=True)
            self.user_occup_embedding_matrix = tf.get_variable("user_occup_embedding_matrix",
                                                               [21, 8],
                                                               trainable=True)

            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.negs_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_negs)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)

        self.user_gender_eb = tf.nn.embedding_lookup(self.user_gender_embedding_matrix, self.user_gender)
        self.user_age_eb = tf.nn.embedding_lookup(self.user_age_embedding_matrix, self.user_age)
        self.user_occup_eb = tf.nn.embedding_lookup(self.user_occup_embedding_matrix, self.user_occup)

        self.pos_item = self.mid_batch_embedded
        self.negs_item = self.negs_batch_embedded
        self.item_his_eb = self.mid_his_batch_embedded * tf.reshape(self.mask, (-1, seq_len, 1))

    def train(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.user_gender: inps[1],
            self.user_age: inps[2],
            self.user_occup: inps[3],
            self.mid_batch_ph: inps[4],
            self.mid_negs: inps[5],
            self.mid_his_batch_ph: inps[6],
            self.mask: inps[7],
            self.lr: inps[8]
        }
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def output_item(self, sess):
        item_embs = sess.run(self.mid_embeddings_var)
        return item_embs

    def output_user(self, sess, inps):
        user_embs = sess.run(self.user_eb, feed_dict={
            self.user_gender: inps[0],
            self.user_age: inps[1],
            self.user_occup: inps[2],
            self.mid_his_batch_ph: inps[3],
            self.mask: inps[4]
        })
        return user_embs

    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)


def _softmax_ce(label_, pred_):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_, logits=pred_)

def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape

class Model_UMI(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len):
        super(Model_UMI, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len,
                                          flag="Model_UMI")
        user_profile = tf.concat([self.user_gender_eb, self.user_age_eb, self.user_occup_eb], axis=-1)  # [bs, emb_size]
        interests = dual_softmax_routing(self.item_his_eb, user_profile, num_interest, embedding_dim, self.mask,
                                         bilinear_type=1)

        user_profile = tf.tile(tf.expand_dims(user_profile, axis=1), [1, num_interest, 1])
        user_attended_profile = _user_profile_interest_attention(interests, user_profile)

        self.user_eb = tf.concat([user_attended_profile, interests], axis=-1)
        with tf.variable_scope("linear_user_net", reuse=tf.AUTO_REUSE):
            linear_units = [int(hidden_size // 2), hidden_size]
            bias = True
            biases_initializer = tf.zeros_initializer() if bias else None

            for i, units in enumerate(linear_units):
                activation_fn = (tf.nn.relu if i < len(linear_units) - 1
                                 else lambda x: x)
                self.user_eb = layers.fully_connected(
                    self.user_eb, units, activation_fn=None,
                    biases_initializer=biases_initializer)
                self.user_eb = activation_fn(self.user_eb)

        self.item_pos_and_negs = tf.concat([tf.expand_dims(self.pos_item, 1), self.negs_item],
                                           axis=1)

        self.readout = hard_attention(self.user_eb, self.item_pos_and_negs)

        # compute loss
        user_item_product = tf.multiply(self.readout, self.item_pos_and_negs)
        self.distance = tf.reduce_sum(user_item_product, 2)
        self.sample_label = tf.reshape(tf.zeros_like(tf.reduce_sum(self.distance, 1), dtype=tf.int64), [-1])

        neg_sampling_loss = _softmax_ce(self.sample_label, self.distance)
        self.loss = tf.reduce_mean(neg_sampling_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
