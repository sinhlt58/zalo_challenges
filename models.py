import os

import tensorflow as tf

from bert import BertModelLayer
from bert import params_from_pretrained_ckpt
from bert import load_stock_weights


def get_rnn_layer(units, dropout=0, rg=1e-4):
    # return (a1...aT, cT) if LSTM (a1...aT, aT, cT)
    # if Bidirectional (a1...aT, a1T, a2T, c1T, c2T)
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True, # return shape (sample, time, units)
            return_state=True, # return last cell
            # dropout=dropout,
            # recurrent_dropout=dropout,
            kernel_regularizer=tf.keras.regularizers.l2(rg),
            recurrent_regularizer=tf.keras.regularizers.l2(rg),
            activity_regularizer=tf.keras.regularizers.l2(rg),
        ),
        merge_mode='concat',
    )

def softmax_with_zero_mask(x):
    x = tf.keras.backend.switch(tf.math.is_nan(x), tf.zeros_like(x), x) # prevent nan values
    x = tf.keras.backend.switch(tf.math.equal(tf.math.exp(x), 1), tf.zeros_like(x), tf.math.exp(x))

    return x / tf.keras.backend.sum(x, axis=-1, keepdims=True)


class EnBertBidaf(tf.keras.Model):

    def __init__(
        self,
        bert_model_path,
        max_length=300,
        q_units=100,
        p_units=200,
    ):
        super(EnBertBidaf, self).__init__()

        # ************** BERT EMBEDDING PART **************
        bert_params = params_from_pretrained_ckpt(bert_model_path)
        self.bert_layer = BertModelLayer.from_params(bert_params, name="bert")
        self.bert_layer.trainable = False

        # # linear transform bert embbedding for question
        # self.linear_q_bert = tf.keras.layers.Dense(q_units, activation="linear")
        # # linear transform bert embbedding for paragraph
        # self.linear_p_bert = tf.keras.layers.Dense(p_units, activation="linear")



        # ******************* BIDAF PART *******************

        # return (a1...aT, cT) if LSTM (a1...aT, aT, cT)
        # if Bidirectional (a1...aT, a1T, a2T, c1T, c2T)
        self.q_lstm1 = get_rnn_layer(q_units)
        self.q_lstm2 = get_rnn_layer(q_units)
        # For weighted average hidden units in question (self attention)
        self.dense_q = tf.keras.layers.Dense(1, activation='linear')

        # return (a1...aT, cT) if LSTM (a1...aT, aT, cT)
        # if Bidirectional (a1...aT, a1T, a2T, c1T, c2T)
        self.p_lstm1 = get_rnn_layer(p_units)
        self.p_lstm2 = get_rnn_layer(p_units)

        # W bilinear for question -> paragraph attention
        self.dense_bilinear = tf.keras.layers.Dense(2*p_units, activation="linear")

        # Dense layers for predicting
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='linear')


    def call(self, input_features, training=True):
        # ************** BERT EMBEDDING PART **************
        questions_texts = input_features[0][:, 0] # (batch, time)
        type_ids = input_features[0][:, 1] # (batch, time)
        # filter padding token
        questions_texts_mask = questions_texts != 0 # (batch, time)

        # print ("questions_texts[0]: ", questions_texts[0])
        # print ("type_ids[0]: ", type_ids[0])
        # print ("questions_texts_mask[0]: ", questions_texts_mask[0])

        # (batch, time, bert hidden)
        bert_hiddens = self.bert_layer([questions_texts, type_ids], mask=questions_texts_mask)

        # (batch, time, 1)
        questions_texts_mask = tf.cast(tf.expand_dims(questions_texts_mask, axis=2), tf.float32)
        q_mask = questions_texts_mask * tf.cast(tf.expand_dims(1 - type_ids, axis=2), tf.float32)
        p_mask = questions_texts_mask * tf.cast(tf.expand_dims(type_ids, axis=2), tf.float32)

        embedd_q = q_mask * bert_hiddens
        embedd_p = p_mask * bert_hiddens

        # NOTE: ATTENTION HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Need to shift the paragraph to right padding in order to use CuDNN
        # https://github.com/tensorflow/tensorflow/issues/30745
        # We shift the paragraphs to right padding
        shifts = -tf.cast(tf.math.reduce_sum(1 - type_ids, axis=1), tf.int32)
        embedd_p = tf.map_fn(
            lambda x: tf.roll(x[0], shift=x[1], axis=0), (embedd_p, shifts), dtype=tf.float32
        )
        p_mask = tf.map_fn(
            lambda x: tf.roll(x[0], shift=x[1], axis=0), (p_mask, shifts), dtype=tf.float32
        )
        # print ("embedd_p[0]: ", embedd_p[0])


        # ******************* BIDAF PART *******************
        # (batch, time, bert hidden) -> (batch, time, p/q_units)
        # embedd_q = self.linear_q_bert(q_bert_hiddens)
        # embedd_p = self.linear_p_bert(p_bert_hiddens)

        q_mask = tf.math.reduce_sum(q_mask, axis=2) == 1
        p_mask = tf.math.reduce_sum(p_mask, axis=2) == 1
        # print ("q_mask[0]: ", q_mask[0])
        # print ("p_mask[0]: ", p_mask[0])

        q_hiddens = self.q_lstm1(embedd_q, mask=q_mask, training=training)[0] # q_hidden.shape (batch, q time, q_units)
        q_hiddens = self.q_lstm2(q_hiddens, mask=q_mask, training=training)[0] # q_hidden.shape (batch, q time, q_units)
        # print ("q_hiddens[0]: ", q_hiddens[0])

        p_hiddens = self.p_lstm1(embedd_p, mask=p_mask, training=training)[0] # p_hiddens.shape (batch, p time, p_units)
        p_hiddens = self.p_lstm2(p_hiddens, mask=p_mask, training=training)[0] # p_hiddens.shape (batch, p time, p_units)
        # print ("p_hiddens[0]: ", p_hiddens[0])
        # print ("p_hiddens[0][10:30]: ", p_hiddens[0][10:30])
        # print ("p_hiddens shape: ", p_hiddens.shape)

        # (batch, q time, 1) -> (batch, q time)
        q_alpha = tf.math.reduce_sum(
            self.dense_q(q_hiddens), axis=2
        )
        # print ("q_alpha[0] before softmax: ", q_alpha[0])
        q_alpha = softmax_with_zero_mask(q_alpha)
        # print ("q_alpha[0] after softmax: ", q_alpha[0])
        # print ("sum: ", tf.math.reduce_sum(q_alpha[0]))
        # print ("q_alpha shape: ", q_alpha.shape)

        # (batch, q time, q_units) x (batch, q time, 1) --> (batch, q_units)
        q_hidden = q_hiddens * tf.expand_dims(q_alpha, axis=2)
        # print ("q_hidden[0]: ", q_hidden[0])
        q_hidden = tf.math.reduce_sum(
            q_hidden, axis=1
        )
        # print ("q_hidden[0]: ", q_hidden[0])

        # calculate attention
        # (batch, q_units) x (q_units, p_units) -> (batch, p_units) -> (batch, 1, p_units)
        q_extend = tf.expand_dims(
            self.dense_bilinear(q_hidden), axis=1
        )
        # print ("q_extend[0]: ", q_extend[0])
        # (batch, p time, p_units) * (batch, 1, p_units) -> (batch, p time, p_units)
        M = p_hiddens * q_extend
        # print ("M[0]: ", M[0])
        # attention weights for paragraph words
        # (batch, p time, p_units) -> (batch, p time)
        alpha = softmax_with_zero_mask(
            tf.math.reduce_sum(M, axis=2)
        )
        # print ("alpha[0]: ", alpha[0])
        # print ("sum alpha[0]: ", tf.math.reduce_sum(alpha[0]))
        # calculate context vector
        # (batch, p time) -> (batch, p time, 1) -> (batch, p time, p_units) -> (batch, p_units)
        dense_input = tf.math.reduce_sum(
            tf.expand_dims(alpha, axis=2) * p_hiddens, axis=1
        )
        # print ("dense_input[0]: ", dense_input[0])

        # (batch, q_units) + (batch, p_units) -> (batch, q_units + p_units)
        # dense_input = tf.concat([q_hidden, dense_input], axis=1)
        # print ("dense_input shape: ", dense_input.shape)

        out = self.dense1(dense_input)
        out = self.dense2(out)

        out = tf.math.reduce_sum(
            out, axis=1
        )

        return out


class AttentiveReader(tf.keras.Model):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        q_units,
        p_units,
        num_rnn_layer=2,
        dropout=0.2,
    ):
        super(AttentiveReader, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True,
        )

        # return (a1...aT, cT) if LSTM (a1...aT, aT, cT)
        # if Bidirectional (a1...aT, a1T, a2T, c1T, c2T)
        self.q_lstm1 = get_rnn_layer(q_units, dropout)
        self.q_lstm2 = get_rnn_layer(q_units, dropout)
        # For weighted average hidden units in question (self attention)
        self.dense_q = tf.keras.layers.Dense(1, activation='linear', use_bias=False)

        # return (a1...aT, cT) if LSTM (a1...aT, aT, cT)
        # if Bidirectional (a1...aT, a1T, a2T, c1T, c2T)
        self.p_lstm1 = get_rnn_layer(p_units, dropout)
        self.p_lstm2 = get_rnn_layer(p_units, dropout)

        # W bilinear for question -> paragraph attention
        self.dense_bilinear = tf.keras.layers.Dense(2*p_units, activation="linear", use_bias=False)

        # Summary weigted question-aware context
        self.p_q_lstm = get_rnn_layer(100, dropout)

        # Dense layers for predicting
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=True):
        question_x = inputs[0]
        paragraph_x = inputs[1]

        embedd_q = self.embedding(question_x) # (batch, q time, embedd_dim)
        embedd_p = self.embedding(paragraph_x) # (batch, p time, embedd_dim)

        # print ("embedd_q shape: ", embedd_q.shape)
        # print ("embedd_q[0]: ", embedd_q[0])
        # print ("embedd_p[0]: ", embedd_p[0])

        mask_q = self.embedding.compute_mask(question_x)
        mask_p = self.embedding.compute_mask(paragraph_x)

        q_hiddens = self.q_lstm1(embedd_q, mask=mask_q, training=training)[0] # q_hidden.shape (batch, q time, q_units)
        q_hiddens = self.q_lstm2(q_hiddens, mask=mask_q, training=training)[0] # q_hidden.shape (batch, q time, q_units)
        # print ("q_hiddens[0]: ", q_hiddens[0])

        # (batch, q time, 1) -> (batch, q time)
        q_alpha = tf.math.reduce_sum(
            self.dense_q(q_hiddens), axis=2
        )
        # print ("q_alpha[0] before softmax: ", q_alpha[0])
        q_alpha = softmax_with_zero_mask(q_alpha)
        # print ("q_alpha[0] after softmax: ", q_alpha[0])
        # print ("sum: ", tf.math.reduce_sum(q_alpha[0]))
        # print ("q_alpha shape: ", q_alpha.shape)

        # (batch, q time, q_units) x (batch, q time, 1) --> (batch, q_units)
        q_hidden = q_hiddens * tf.expand_dims(q_alpha, axis=2)
        # print ("q_hidden[0]: ", q_hidden[0])
        q_hidden = tf.math.reduce_sum(
            q_hidden, axis=1
        )
        # print ("q_hidden[0]: ", q_hidden[0])

        p_hiddens = self.p_lstm1(embedd_p, mask=mask_p, training=training)[0] # p_hiddens.shape (batch, p time, p_units)
        p_hiddens = self.p_lstm2(p_hiddens, mask=mask_p, training=training)[0] # p_hiddens.shape (batch, p time, p_units)
        # print ("p_hiddens[0]: ", p_hiddens[0])
        # print ("p_hiddens shape: ", p_hiddens.shape)

        # calculate attention
        # (batch, q_units) x (q_units, p_units) -> (batch, p_units) -> (batch, 1, p_units)
        q_extend = tf.expand_dims(
            self.dense_bilinear(q_hidden), axis=1
        )
        # print ("q_extend[0]: ", q_extend[0])
        # (batch, p time, p_units) * (batch, 1, p_units) -> (batch, p time, p_units)
        M = p_hiddens * q_extend
        # print ("M[0]: ", M[0])
        # attention weights for paragraph words
        # (batch, p time, p_units) -> (batch, p time)
        alpha = softmax_with_zero_mask(
            tf.math.reduce_sum(M, axis=2)
        )
        # print ("alpha[0]: ", alpha[0])
        # print ("sum alpha[0]: ", tf.math.reduce_sum(alpha[0]))
        # calculate context vector
        # (batch, p time) -> (batch, p time, 1) -> (batch, p time, p_units)
        weighted_p_hiddens = tf.expand_dims(alpha, axis=2) * p_hiddens

        # p_q_hiddens.shape (batch, p time, p_units)
        p_q_hiddens = self.p_q_lstm(weighted_p_hiddens, mask=mask_p, training=training)[0]
        # shape (batch, p_units)
        dense_input = tf.math.reduce_sum(
            p_q_hiddens, axis=1
        )
        # print ("dense_input[0]: ", dense_input[0])

        # (batch, q_units) + (batch, p_units) -> (batch, q_units + p_units)
        # dense_input = tf.concat([q_hidden, dense_input], axis=1)
        # print ("dense_input shape: ", dense_input.shape)

        out = self.dense1(dense_input)
        out = self.dense2(out)

        out = tf.math.reduce_sum(out, axis=1)

        return out


def get_model(
    lang,
    model_type,
    bert_model_path,
    max_length=300,
    num_feature=2,
    saved_epoch_path=None,
    configs=None,
):

    if model_type == "vi_attentive_reader":
        question_size = configs["question_size"]
        text_size = configs["text_size"]

        question_input = tf.keras.layers.Input(shape=(question_size))
        text_input = tf.keras.layers.Input(shape=(text_size))
        inputs = [question_input, text_input]

        attentive_reader = AttentiveReader(
            vocab_size=configs["vocab_size"],
            embedding_dim=200,
            q_units=200,
            p_units=200,
            num_rnn_layer=2,
        )
        output = attentive_reader(inputs)

        model = tf.keras.Model(inputs=inputs, outputs=output)

        if saved_epoch_path:
            # load the saved model
            # TODO: we will not save bert weights later
            print ("Loading saved_epoch_path: {}".format(
                saved_epoch_path
            ))
            model.load_weights(saved_epoch_path)

        return model

    elif model_type == "en_bert_bidaf":
        input_features = [tf.keras.layers.Input(shape=(num_feature, max_length))]

        bert_bidaf = EnBertBidaf(
            bert_model_path=bert_model_path,
            max_length=max_length,
        )
        output = bert_bidaf(input_features)

        model = tf.keras.Model(inputs=input_features, outputs=output)

        if saved_epoch_path:
            # load the saved model
            # TODO: we will not save bert weights later
            print ("Loading saved_epoch_path: {}".format(
                saved_epoch_path
            ))
            model.load_weights(saved_epoch_path)
        else:
            # load weights for bert model
            weights_file = "{}/bert_model.ckpt".format(bert_model_path)
            print ("Loading bert weights_file: {}".format(
                weights_file
            ))
            load_stock_weights(bert_bidaf.bert_layer, weights_file)

        return model
