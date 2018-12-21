from networks import Generator, Discriminator
from ops import Hinge_Loss, MSE
import tensorflow as tf
from utils import read_crop_data
import numpy as np
from PIL import Image

#Paper: CGANS WITH PROJECTION DISCRIMINATOR
#Paper: Residual Dense Network for Image Super-Resolution

BATCH_SIZE = 16
MAX_ITERATION = 600000
TRAINING_SET_PATH = "./TrainingSet/"
LAMBDA = 100
SAVE_MODEL = "./save_para/"
RESULTS = "./results/"



def train():
    RDN = Generator("RDN")
    D = Discriminator("discriminator")
    HR = tf.placeholder(tf.float32, [None, 96, 96, 3])
    LR = tf.placeholder(tf.float32, [None, 24, 24, 3])
    SR = RDN(LR)
    fake_logits = D(SR, LR)
    real_logits = D(HR, LR)
    D_loss, G_loss = Hinge_Loss(fake_logits, real_logits)
    G_loss += MSE(SR, HR) * LAMBDA
    itr = tf.Variable(MAX_ITERATION, dtype=tf.int32, trainable=False)
    learning_rate = tf.Variable(2e-4, trainable=False)
    op_sub = tf.assign_sub(itr, 1)
    D_opt = tf.train.AdamOptimizer(learning_rate, beta1=0., beta2=0.9).minimize(D_loss, var_list=D.var_list())
    with tf.control_dependencies([op_sub]):
        G_opt = tf.train.AdamOptimizer(learning_rate, beta1=0., beta2=0.9).minimize(G_loss, var_list=RDN.var_list())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    while True:
        HR_data, LR_data = read_crop_data(TRAINING_SET_PATH, BATCH_SIZE, [96, 96, 3], 4)
        sess.run(D_opt, feed_dict={HR: HR_data, LR: LR_data})
        [_, iteration] = sess.run([G_opt, itr], feed_dict={HR: HR_data, LR: LR_data})
        iteration = MAX_ITERATION - iteration
        if iteration < MAX_ITERATION // 2:
            learning_rate = learning_rate * (iteration * 2 / MAX_ITERATION)
        if iteration % 10 == 0:
            [D_LOSS, G_LOSS, LEARNING_RATE, img] = sess.run([D_loss, G_loss, learning_rate, SR], feed_dict={HR: HR_data, LR: LR_data})
            output = (np.concatenate((HR_data[0, :, :, :], img[0, :, :, :]), axis=1) + 1) * 127.5
            Image.fromarray(np.uint8(output)).save(RESULTS+str(iteration)+".jpg")
            print("Iteration: %d, D_loss: %f, G_loss: %f, LearningRate: %f"%(iteration, D_LOSS, G_LOSS, LEARNING_RATE))
        if iteration % 500 == 0:
            saver.save(sess, SAVE_MODEL + "model.ckpt")

train()
