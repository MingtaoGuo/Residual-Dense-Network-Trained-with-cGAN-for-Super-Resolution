from ops import *

class Generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, G_0=64, G=32, D=20, C_nums=6):
        #Section 5.3 for configuration
        #Grouth rate: G_0, G, The number of RDB: D
        with tf.variable_scope(self.name):
            inputs = relu(conv("conv1", inputs, 3, G_0, 1))
            F_1 = tf.identity(inputs)
            inputs = relu(conv("conv2", inputs, 3, G_0, 1))

            inputs = RDB("0", inputs, C_nums=C_nums, G=G, G_0=G_0)
            temp = tf.identity(inputs)
            for i in range(1, D):
                inputs = RDB(str(i), inputs, C_nums=C_nums, G=G, G_0=G_0)
                temp = tf.concat([inputs, temp], axis=-1)
            inputs = relu(conv("conv3", temp, 1, G_0, 1))
            F_GF = relu(conv("conv4", inputs, 3, G_0, 1))
            F_DF = F_GF + F_1
            inputs = Upscale(F_DF, 2)
            inputs = relu(conv("Up_conv1", inputs, 3, G, 1))
            inputs = Upscale(inputs, 2)
            inputs = conv("Up_conv2", inputs, 3, 3, 1)
        return tf.nn.tanh(inputs)

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

class Discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, y):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = ResBlock("Res1", inputs, 3, 64)
            inputs = ResBlock("Res2", inputs, 3, 64, False)
            inputs = ResBlock("Res3", inputs, 3, 128)
            inputs = ResBlock("Res4", inputs, 3, 128, False)
            x = Inner_product(inputs, y)
            inputs = ResBlock("Res5", inputs, 3, 128)
            inputs = ResBlock("Res6", inputs, 3, 256)#256
            inputs = ResBlock("Res7", inputs, 3, 512)#512
            inputs = ResBlock("Res8", inputs, 3, 1024)#1024
            inputs = ResBlock("Res9", inputs, 3, 1024, False)#1024
            inputs = relu(inputs)
            inputs = global_sum_pooling(inputs)
            inputs = Linear("Linear", inputs, 1024, 1) + x
        return inputs

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)