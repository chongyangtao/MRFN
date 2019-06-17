import os, sys
import tensorflow as tf
import numpy as np
from modules import multihead_attention, conv, batch_coattention_nnsubmulti


class model(object):
    def __init__(self, batch, FLAGS, embed_dim, vocab_size, char_dim, char_size, rnn_dim, max_turn, max_word_len, max_char_len, \
                            pretrained_word_embeddings=None, pretrained_char_embeddings=None):

        self.embed_dim = embed_dim
        self.char_dim = char_dim
        self.rnn_dim = rnn_dim
        self.max_turn = max_turn
        self.max_word_len = max_word_len
        self.max_char_len = max_char_len

        self.context, self.context_mask, self.context_len, \
            self.response, self.response_mask, self.response_len, \
                self.char_context, self.char_context_mask, self.char_context_len, \
                    self.char_response, self.char_response_mask, self.char_response_len, self.target = batch.get_next()

        self.context_mask = tf.to_float(self.context_mask)
        self.response_mask = tf.to_float(self.response_mask)

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        self.y_pred = 0.0
        self.loss = 0.0
        self.loss_list = []


        self.expand_response_len = tf.tile(tf.expand_dims(self.response_len, 1), [1, max_turn])
        self.expand_response_len = tf.reshape(self.expand_response_len, [-1])
        self.expand_response_mask = tf.tile(tf.expand_dims(self.response_mask, 1), [1, max_turn, 1]) 
        self.expand_response_mask = tf.reshape(self.expand_response_mask, [-1, max_word_len])

        self.parall_context_len = tf.reshape(self.context_len, [-1])
        self.parall_context_mask = tf.reshape(self.context_mask, [-1, max_word_len])

        self.all_utterance_mask = tf.unstack(self.context_mask, num=max_turn, axis=1)
        self.concat_context_mask = tf.concat(self.all_utterance_mask, axis=1)

        # character-based word representation
        if FLAGS.use_char:
            conv_dim = FLAGS.char_hid
            kernels = [5]
            with tf.variable_scope("char_embeddings"):
                char_embeddings = tf.get_variable('char_embeddings_v', shape=(char_size, char_dim), dtype=tf.float32, trainable=True)
                if pretrained_char_embeddings is not None:
                    self.char_embeddings_init = char_embeddings.assign(pretrained_char_embeddings)
                response_char_embeddings = tf.nn.embedding_lookup(char_embeddings, self.char_response) 
                response_char_embeddings = tf.reshape(response_char_embeddings, [-1, max_char_len, char_dim]) 
                response_char_embeddings = conv(response_char_embeddings, conv_dim, kernel_size=kernels, bias=True, activation=tf.nn.relu, name="char_conv", isNormalize=False, reuse=None)
                response_char_embeddings = tf.reduce_max(response_char_embeddings, axis=1)
                response_char_embeddings = tf.reshape(response_char_embeddings, [-1, max_word_len, response_char_embeddings.get_shape().as_list()[-1]]) 

                response_char_embeddings = tf.layers.dropout(response_char_embeddings, rate=1-self.dropout_keep_prob)
                self.expand_response_char_embeddings = tf.tile(tf.expand_dims(response_char_embeddings, 1), [1, max_turn, 1, 1]) 
                self.expand_response_char_embeddings = tf.reshape(self.expand_response_char_embeddings, [-1, max_word_len, conv_dim])  

                context_char_embeddings = tf.nn.embedding_lookup(char_embeddings, self.char_context) # [batch, max_turn, max_word_len, max_char_len, char_dim]
                
                # context_char_embeddings = tf.reshape(context_char_embeddings, [-1, max_char_len, char_dim])
                # context_char_embeddings = conv(context_char_embeddings, conv_dim, kernel_size=kernels, bias=True, activation=tf.nn.relu, name="char_conv", isNormalize=False, reuse=True)
                # context_char_embeddings = tf.reduce_max(context_char_embeddings, axis = 1)
                # self.parall_context_char_embeddings = tf.reshape(context_char_embeddings, [-1, max_word_len, conv_dim*len(kernels)]) 
                # self.parall_context_char_embeddings = tf.layers.dropout(self.parall_context_char_embeddings, rate=1-self.dropout_keep_prob)

                cont_char_embeddings = []
                for k, utt_char_emb in enumerate(tf.unstack(context_char_embeddings, axis=1)):
                    utt_char_embeddings = tf.reshape(utt_char_emb, [-1, max_char_len, char_dim])
                    utt_char_embeddings = conv(utt_char_embeddings, conv_dim, kernel_size=kernels, bias=True, activation=tf.nn.relu, name="char_conv", isNormalize=False, reuse=True)
                    utt_char_embeddings = tf.reduce_max(utt_char_embeddings, axis=1)
                    utt_char_embeddings = tf.reshape(utt_char_embeddings, [-1, max_word_len, utt_char_embeddings.get_shape().as_list()[-1]]) 
                    cont_char_embeddings.append(utt_char_embeddings)
                context_char_embeddings = tf.stack(cont_char_embeddings, axis=1)  
                self.parall_context_char_embeddings = tf.reshape(context_char_embeddings, [-1, max_word_len, conv_dim*len(kernels)])

            char_interaction = self.interaction_matching_batch(self.parall_context_char_embeddings, self.expand_response_char_embeddings, conv_dim, scope='char_interaction_matching')
            loss_char = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=char_interaction))
            self.loss += loss_char
            self.y_pred += tf.nn.softmax(char_interaction)
            self.loss_list.append(loss_char)


        with tf.variable_scope("word_embeddings"):
            word_embeddings = tf.get_variable('word_embeddings_v', shape=(vocab_size, embed_dim), dtype=tf.float32, trainable=True) 
            if pretrained_word_embeddings is not None:
                self.embedding_init = word_embeddings.assign(pretrained_word_embeddings)

            self.context_embeddings = tf.nn.embedding_lookup(word_embeddings, self.context)
            self.response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response)
            self.context_embeddings = tf.layers.dropout(self.context_embeddings, rate=1-self.dropout_keep_prob)
            self.response_embeddings = tf.layers.dropout(self.response_embeddings, rate=1-self.dropout_keep_prob)

            self.parall_context_embeddings = tf.reshape(self.context_embeddings, [-1, max_word_len, embed_dim]) 
            self.all_utterance_embeddings = tf.unstack(self.context_embeddings, num=max_turn, axis=1)

            self.expand_response_embeddings = tf.tile(tf.expand_dims(self.response_embeddings, 1), [1, max_turn, 1, 1]) 
            self.expand_response_embeddings = tf.reshape(self.expand_response_embeddings, [-1, max_word_len, embed_dim])  


        # word representation
        if FLAGS.use_word:
            word_interaction = self.interaction_matching_batch(self.parall_context_embeddings, self.expand_response_embeddings, embed_dim, scope='word_interaction_matching')
            loss_word = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=word_interaction))
            self.loss += loss_word
            self.y_pred += tf.nn.softmax(word_interaction)
            self.loss_list.append(loss_word)


        # sequential representation
        if FLAGS.use_seq:
            with tf.variable_scope("RNN_embeddings"):
                sentence_gru_cell = tf.contrib.rnn.GRUCell(rnn_dim)
                with tf.variable_scope('sentence_gru'):
                    self.response_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_gru_cell, self.response_embeddings, sequence_length=self.response_len, dtype=tf.float32)
                with tf.variable_scope('sentence_gru', reuse=True):
                    self.context_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_gru_cell, self.parall_context_embeddings, sequence_length=self.parall_context_len, dtype=tf.float32)

            self.expand_response_GRU_embeddings = tf.tile(tf.expand_dims(self.response_GRU_embeddings, 1), [1, max_turn, 1, 1]) 
            self.expand_response_GRU_embeddings = tf.reshape(self.expand_response_GRU_embeddings, [-1, max_word_len, rnn_dim])

            seg_interaction = self.interaction_matching_batch(self.context_GRU_embeddings, self.expand_response_GRU_embeddings, rnn_dim, scope='seg_interaction_matching')
            loss_seg = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=seg_interaction))
            self.loss += loss_seg
            self.y_pred += tf.nn.softmax(seg_interaction)
            self.loss_list.append(loss_seg)

        # local representation
        if FLAGS.use_conv:
            conv_dim = 50
            kernels = [1,2,3,4]
            with tf.variable_scope("conv_embeddings"):
                self.response_conv_embeddings = conv(self.response_embeddings, conv_dim, kernel_size=kernels, bias=True, activation=tf.nn.relu, isNormalize=True, reuse=False)
                self.context_conv_embeddings = conv(self.parall_context_embeddings, conv_dim, kernel_size=kernels, bias=True, activation=tf.nn.relu, isNormalize=True, reuse=True)

            self.expand_response_conv_embeddings = tf.tile(tf.expand_dims(self.response_conv_embeddings, 1), [1, max_turn, 1, 1]) 
            self.expand_response_conv_embeddings = tf.reshape(self.expand_response_conv_embeddings, [-1, max_word_len, conv_dim*len(kernels)])  
            conv_interaction = self.interaction_matching_batch(self.context_conv_embeddings, self.expand_response_conv_embeddings, conv_dim*len(kernels), scope='conv_interaction_matching')
            loss_conv = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=conv_interaction))
            self.loss += loss_conv
            self.y_pred += tf.nn.softmax(conv_interaction)
            self.loss_list.append(loss_conv)

        # self attention
        if FLAGS.use_self:
            with tf.variable_scope("self_att_embeddings"): 
                self.response_self_att_embeddings = multihead_attention(self.response_embeddings, self.response_embeddings, embed_dim, reuse=False)
                self.context_self_att_embeddings = multihead_attention(self.parall_context_embeddings, self.parall_context_embeddings, embed_dim, reuse=True)
            
            self.expand_response_self_att_embeddings = tf.tile(tf.expand_dims(self.response_self_att_embeddings, 1), [1, max_turn, 1, 1]) 
            self.expand_response_self_att_embeddings = tf.reshape(self.expand_response_self_att_embeddings, [-1, max_word_len, embed_dim])  
            self_att_interaction = self.interaction_matching_batch(self.context_self_att_embeddings, self.expand_response_self_att_embeddings, 
                                                                                    embed_dim, scope='self_att_interaction_matching')
            self.y_pred +=  tf.nn.softmax(self_att_interaction)
            loss_self = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self_att_interaction)) 
            self.loss += loss_self
            self.loss_list.append(loss_self)


        # cross attention 
        if FLAGS.use_cross:
            with tf.variable_scope("cross_att_embeddings"): 
                expand_response_embeddings = tf.tile(tf.expand_dims(self.response_embeddings, 1), [1, self.max_turn, 1, 1])
                expand_response_embeddings = tf.reshape(expand_response_embeddings, [-1, max_word_len, embed_dim])       
                self.context_cross_att_embeddings = multihead_attention(self.parall_context_embeddings, expand_response_embeddings, rnn_dim, reuse=False) 

                self.response_cross_att_embeddings = []                
                for k, utterance_embeddings in enumerate(self.all_utterance_embeddings):
                    response_cross_att_embedding = multihead_attention(self.response_embeddings, utterance_embeddings, rnn_dim, reuse=True) 
                    self.response_cross_att_embeddings.append(response_cross_att_embedding)
                self.response_cross_att_embeddings = tf.stack(self.response_cross_att_embeddings, axis=1) 
                self.response_cross_att_embeddings = tf.reshape(self.response_cross_att_embeddings, [-1, max_word_len, embed_dim])        

            logits = self.interaction_matching_batch(self.context_cross_att_embeddings, self.response_cross_att_embeddings, embed_dim, scope='cross_att_interaction_matching')
            loss_cross = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=logits))
            self.loss += loss_cross
            self.y_pred += tf.nn.softmax(logits)
            self.loss_list.append(loss_cross)


        with tf.name_scope("accuracy"):
            self.correct = tf.equal(tf.cast(tf.argmax(self.y_pred, axis=1), tf.int32), tf.to_int32(self.target))  
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, 'float'))


    def interaction_matching_batch(self, context_embeddings, response_embeddings, rnn_dim, scope="interaction_matching", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            res_coatt = batch_coattention_nnsubmulti(context_embeddings, response_embeddings, tf.to_float(self.parall_context_mask), scope='%s_res'%scope)  
            res_coatt = tf.layers.dropout(res_coatt, rate=1-self.dropout_keep_prob)

            res_coatt_cell = tf.contrib.rnn.GRUCell(rnn_dim)
            with tf.variable_scope('res_att_gru'):
                res_hiddens, res_final = tf.nn.dynamic_rnn(res_coatt_cell, res_coatt, dtype=tf.float32) 

            res_feature = res_final
            res_feature = tf.reshape(res_feature, [-1, self.max_turn, rnn_dim])

            final_gru_cell = tf.contrib.rnn.GRUCell(rnn_dim)
            _, last_hidden = tf.nn.dynamic_rnn(final_gru_cell, res_feature, dtype=tf.float32, scope='final_GRU')  
            logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')
            return logits


