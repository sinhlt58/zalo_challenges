import tensorflow as tf


class AttentiveReader(tf.keras.Model):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        q_units,
        p_units,
        num_rnn_layer=2,
    ):
        super(AttentiveReader, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True,
        )

        # return (a1...aT, cT) if LSTM (a1...aT, aT, cT)
        # if Bidirectional (a1...aT, a1T, a2T, c1T, c2T)
        self.q_lstm1 = self._get_rnn_layer(q_units)
        self.q_lstm2 = self._get_rnn_layer(q_units)
        # For weighted average hidden units in question (self attention)
        self.dense_q = tf.keras.layers.Dense(1, activation='linear')

        # return (a1...aT, cT) if LSTM (a1...aT, aT, cT)
        # if Bidirectional (a1...aT, a1T, a2T, c1T, c2T)
        self.p_lstm1 = self._get_rnn_layer(p_units)
        self.p_lstm2 = self._get_rnn_layer(p_units)

        # W bilinear for question -> paragraph attention
        self.dense_bilinear = tf.keras.layers.Dense(2*p_units, activation="linear")

        # Dense layers for predicting
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, question_x, paragraph_x):
        embedd_q = self.embedding(question_x) # (batch, q time, embedd_dim)
        embedd_p = self.embedding(paragraph_x) # (batch, p time, embedd_dim)

        # print ("embedd_q shape: ", embedd_q.shape)
        # print ("embedd_q[0]: ", embedd_q[0])

        mask_q = self.embedding.compute_mask(question_x)
        mask_p = self.embedding.compute_mask(paragraph_x)

        q_hiddens = self.q_lstm1(embedd_q, mask=mask_q)[0] # q_hidden.shape (batch, q time, q_units)
        q_hiddens = self.q_lstm2(q_hiddens, mask=mask_q)[0] # q_hidden.shape (batch, q time, q_units)
        # print ("q_hiddens[0]: ", q_hiddens[0])

        # (batch, q time, 1) -> (batch, q time)
        q_alpha = tf.math.reduce_sum(
            self.dense_q(q_hiddens), axis=2
        )
        # print ("q_alpha[0] before softmax: ", q_alpha[0])
        q_alpha = self._softmax_with_zero_mask(q_alpha)
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

        p_hiddens = self.p_lstm1(embedd_p, mask=mask_p)[0] # p_hiddens.shape (batch, p time, p_units)
        p_hiddens = self.p_lstm2(p_hiddens, mask=mask_p)[0] # p_hiddens.shape (batch, p time, p_units)
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
        alpha = self._softmax_with_zero_mask(
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

        return out

    def _get_rnn_layer(self, units):
        # return (a1...aT, cT) if LSTM (a1...aT, aT, cT)
        # if Bidirectional (a1...aT, a1T, a2T, c1T, c2T)
        return tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=units,
                recurrent_initializer='glorot_uniform',
                return_sequences=True, # return shape (sample, time, units)
                return_state=True, # return last cell
            ),
            merge_mode='concat',
        )

    def _softmax_with_zero_mask(self, x):
        x = tf.keras.backend.switch(tf.math.is_nan(x), tf.zeros_like(x), x) # prevent nan values
        x = tf.keras.backend.switch(tf.math.equal(tf.math.exp(x), 1), tf.zeros_like(x), tf.math.exp(x))

        return x / tf.keras.backend.sum(x, axis=-1, keepdims=True)
