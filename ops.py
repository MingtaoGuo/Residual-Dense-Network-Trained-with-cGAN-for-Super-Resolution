import tensorflow as tf



def batchnorm(x, train_phase, scope_bn):
    #Batch Normalization
    #Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    with tf.variable_scope(scope_bn):
        beta = tf.get_variable(name=scope_bn + 'beta', shape=[x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
        gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def _l2normalize(v, eps=1e-12):
    return v / tf.sqrt(tf.reduce_sum(tf.square(v)) + eps)


def max_singular_value(W, u, Ip=1):
    _u = u
    _v = 0
    for _ in range(Ip):
        _v = _l2normalize(tf.matmul(_u, W), eps=1e-12)
        _u = _l2normalize(tf.matmul(_v, W, transpose_b=True), eps=1e-12)
    _v = tf.stop_gradient(_v)
    _u = tf.stop_gradient(_u)
    sigma = tf.reduce_sum(tf.matmul(_u, W) * _v)
    return sigma, _u, _v

def spectral_normalization(name, W, Ip=1):
    u = tf.get_variable(name + "_u", [1, W.shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)  # 1 x ch
    W_mat = tf.transpose(tf.reshape(W, [-1, W.shape[-1]]))
    sigma, _u, _ = max_singular_value(W_mat, u, Ip)
    with tf.control_dependencies([tf.assign(u, _u)]):
        W_sn = W / sigma
    return W_sn


def conv(name, inputs, k_size, nums_out, strides, is_sn=False):
    nums_in = int(inputs.shape[-1])
    kernel = tf.get_variable(name+"W", [k_size, k_size, nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
    bias = tf.get_variable(name+"B", [nums_out], initializer=tf.constant_initializer(0.))
    if is_sn:
        return tf.nn.conv2d(inputs, spectral_normalization(name, kernel), [1, strides, strides, 1], "SAME") + bias
    else:
        return tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], "SAME") + bias

def relu(inputs):
    return tf.nn.relu(inputs)

def leaky_relu(inputs, slope=0.2):
    return tf.maximum(inputs, slope * inputs)

def RDB(name, inputs, C_nums, G, G_0):
    #Paper: Figure 3.
    with tf.variable_scope("RDB_"+name):
        temp = tf.identity(inputs)
        for i in range(C_nums):
            x = conv("conv1_" + str(i), inputs, 3, G, 1)
            x = relu(x)
            inputs = tf.concat([inputs, x], axis=-1)
        inputs = conv("conv", inputs, 1, G_0, 1)
        inputs = temp + inputs
    return inputs

def Upscale(inputs, factor):
    B = tf.shape(inputs)[0]
    H = tf.shape(inputs)[1]
    W = tf.shape(inputs)[2]
    nums_in = int(inputs.shape[-1])
    nums_out = nums_in // factor ** 2
    inputs = tf.split(inputs, num_or_size_splits=nums_out, axis=-1)
    output = 0
    for idx, split in enumerate(inputs):
        temp = tf.reshape(split, [B, H, W, factor, factor])
        temp = tf.transpose(temp, perm=[0, 1, 4, 2, 3])
        temp = tf.reshape(temp, [B, H * factor, W * factor, 1])
        if idx == 0:
            output = temp
        else:
            output = tf.concat([output, temp], axis=-1)
    return output

def Linear(name, inputs, nums_in, nums_out, is_sn=True):
    W = tf.get_variable("W_" + name, [nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable("B_" + name, [nums_out], initializer=tf.constant_initializer([0.]))
    if is_sn:
        return tf.matmul(inputs, spectral_normalization(name, W)) + b
    else:
        return tf.matmul(inputs, W) + b

def avg_pool(inputs, k_size=3, strides=2, padding="SAME"):
    return tf.nn.avg_pool(inputs, [1, k_size, k_size, 1], [1, strides, strides, 1], padding)

def ResBlock(name, inputs, k_size, nums_out, is_down=True):
    #inputs: B x H x W x C_in
    with tf.variable_scope(name):
        temp = inputs
        inputs = relu(inputs)
        inputs = conv("conv1", inputs, k_size, nums_out, 1, True)  # inputs: B x H/2 x W/2 x C_out
        inputs = relu(inputs)
        inputs = conv("conv2", inputs, k_size, nums_out, 1, True)  # inputs: B x H/2 x W/2 x C_out
        if is_down:
            inputs = avg_pool(inputs)
            down_sampling = conv("down_sampling_" + name, temp, 1, nums_out, 1, True)  # down_sampling: B x H x W x C_out
            down_sampling = avg_pool(down_sampling)
            outputs = inputs + down_sampling
        else:
            outputs = inputs + temp
    return outputs

def ResBlock0(name, inputs, k_size, nums_out, is_down=True):
    #inputs: B x H x W x C_in
    with tf.variable_scope(name):
        temp = inputs
        inputs = conv("conv1", inputs, k_size, nums_out, 1, True)  # inputs: B x H/2 x W/2 x C_out
        inputs = relu(inputs)
        inputs = conv("conv2", inputs, k_size, nums_out, 1, True)  # inputs: B x H/2 x W/2 x C_out
        inputs = relu(inputs)
        if is_down:
            inputs = avg_pool(inputs)
            down_sampling = conv("down_sampling_" + name, temp, 1, nums_out, 1, True)  # down_sampling: B x H x W x C_out
            down_sampling = avg_pool(down_sampling)
            outputs = inputs + down_sampling
        else:
            outputs = inputs + temp
    return outputs

def Inner_product(inputs, y):
    with tf.variable_scope("IP"):
        inputs = conv("conv", inputs, 3, 3, 1, True)
    inputs = tf.reduce_sum(inputs * y, axis=[1, 2, 3])
    return inputs

def global_sum_pooling(inputs):
    return tf.reduce_sum(inputs, axis=[1, 2])

def Hinge_Loss(fake_logits, real_logits):
    D_loss = tf.reduce_mean(tf.maximum(0., 1 - real_logits)) + \
             tf.reduce_mean(tf.maximum(0., 1 + fake_logits))
    G_loss = -tf.reduce_mean(fake_logits)
    return D_loss, G_loss

def MSE(a, b):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(a - b), axis=[1, 2, 3]))




