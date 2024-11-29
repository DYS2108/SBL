from __future__ import division
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.sparse
import math
import time
import h5py

def get_model_baselearning(x, gcn_adj,re_use, is_train_big):

    with tf.variable_scope("baselearning_model",reuse=re_use):

        feature0 = tf.layers.dense(x, units=512,name="den_0",activation=tf.nn.tanh,trainable=is_train_big)
        feature0_1 = tf.layers.dense(feature0, units=1000, name="den_1",activation=tf.nn.tanh,trainable=is_train_big)

        g_conv0 = feature0_1
        weight1 = tf.get_variable(name="bweight1", shape=[1000,1000], initializer=tf.truncated_normal_initializer(mean=0,stddev=1),
                                  trainable=is_train_big)
        g_conv1 = tf.matmul(tf.sparse_tensor_dense_matmul(gcn_adj, g_conv0), weight1)

        weight2 = tf.get_variable(name="bweight2", shape=[1000, 1000],initializer=tf.truncated_normal_initializer(mean=0, stddev=1),
                                  trainable=is_train_big)
        g_conv2 = tf.matmul(tf.sparse_tensor_dense_matmul(gcn_adj, g_conv1), weight2)

        weight3 = tf.get_variable(name="bweight3", shape=[1000, 1000],initializer=tf.truncated_normal_initializer(mean=0, stddev=1),
                                  trainable=is_train_big)
        g_conv3 = tf.matmul(tf.sparse_tensor_dense_matmul(gcn_adj, g_conv2), weight3)

        weight4 = tf.get_variable(name="bweight4", shape=[1000, 1000],initializer=tf.truncated_normal_initializer(mean=0, stddev=1),
                                  trainable=is_train_big)
        g_conv4 = tf.matmul(tf.sparse_tensor_dense_matmul(gcn_adj, g_conv3), weight4)

        weight5 = tf.get_variable(name="bweight5", shape=[1000, 1000],initializer=tf.truncated_normal_initializer(mean=0, stddev=1),
                                  trainable=is_train_big)
        g_conv5 = tf.matmul(tf.sparse_tensor_dense_matmul(gcn_adj, g_conv4), weight5)

        re_base = g_conv5

        weight_select = tf.get_variable(name="weights", shape=[1000, 16], initializer=tf.truncated_normal_initializer(mean=0,stddev=1))
        weight_select = tf.nn.softmax(weight_select, axis=0)

        re_bases = tf.nn.l2_normalize(tf.matmul(re_base, weight_select), axis=0)


        return re_base, re_bases

