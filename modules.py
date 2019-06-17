import tensorflow as tf

initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)

def linear(args, output_size, bias=False):
    total_arg_size = 0
    shapes = [arg.get_shape() for arg in args]
    for shape in shapes:
        if shape[-1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[-1].value
    dtype = args[0].dtype

    _scope = tf.get_variable_scope()
    with tf.variable_scope(_scope) as outer_scope:
        W = tf.get_variable('W', [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
        logits = tf.einsum('aij,jk->aik', args[0], W) #TODO: check
    else:
        logits = tf.einsum('aij,jk->aik', tf.concat(args, -1), W)
    if not bias:
        return logits
    b = tf.get_variable('b', [output_size], dtype=dtype, initializer=tf.constant_initializer(0.0, dtype=dtype))
    return tf.nn.bias_add(logits, b)


def masked_softmax(scores, mask):
    numerator = tf.exp(tf.subtract(scores, tf.reduce_max(scores, 2, keep_dims=True))) * tf.expand_dims(mask, axis=1)
    denominator = tf.reduce_sum(numerator, 2, keep_dims=True)
    weights = tf.div(numerator + 1e-5 / mask.get_shape()[-1].value, denominator + 1e-5)
    return weights


def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))

        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def conv(inputs, output_size, kernel_size = [1,2,3,4], bias = None, activation = None, name = "conv", isNormalize= False, reuse = None):
    with tf.variable_scope(name, reuse = reuse):
        conv_features = []
        shapes = inputs.shape.as_list()
        for k in kernel_size:
            filter_shape = [k, shapes[-1], output_size]
            bias_shape = [1,1,output_size]
            strides = 1
            kernel_ = tf.get_variable("kernel_%s"%k,
                            filter_shape,
                            dtype = tf.float32,
                            regularizer=regularizer,
                            initializer = initializer())
            feature = tf.nn.conv1d(inputs, kernel_, strides, "SAME")
            if bias:
                feature += tf.get_variable("bias_%s"%k,
                            bias_shape,
                            regularizer=regularizer,
                            initializer = tf.zeros_initializer())
            if activation is not None:
                feature = activation(feature)
            conv_features.append(feature)
        output = tf.concat(conv_features, axis=-1) # [, max_len, ]
        if isNormalize:
            output = normalize(output, 1e-8, "normalize", reuse) 
        return output


def multihead_attention(queries, keys, num_units=None, num_heads=8, dropout_rate=0,
                            is_training=True, causality=False, scope="multihead_attention", reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu, name="dense_q")  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu, name="dense_k")  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu, name="dense_v")  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def batch_coattention_nnsubmulti(utterance, response, utterance_mask, scope="co_attention", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        dim = utterance.get_shape().as_list()[-1]
        weight = tf.get_variable('Weight', shape=[dim, dim], dtype=tf.float32)
        e_utterance = tf.einsum('aij,jk->aik', utterance, weight)
        a_matrix = tf.matmul(response, tf.transpose(e_utterance, perm=[0,2,1]))
        reponse_atten = tf.matmul(masked_softmax(a_matrix, utterance_mask), utterance)
        feature_mul = tf.multiply(reponse_atten, response)
        feature_sub = tf.subtract(reponse_atten, response)
        feature_last = tf.layers.dense(tf.concat([feature_mul, feature_sub], axis=-1), dim, use_bias=True, activation=tf.nn.relu, reuse=reuse) 
    return feature_last



