from __future__ import division
import tensorflow as tf
import numpy as np
import math
import time
import h5py
import argparse
from model import *
from utils import *
import os
import random
import scipy.io as sio
import scipy.sparse as scisp
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist


sess = tf.InteractiveSession()

parser = argparse.ArgumentParser()
parser.add_argument('--architecture', type=int, default=0)
parser.add_argument('--dataset_path', type=str) # the path saving training and testing data
parser.add_argument('--results_path', type=str) # the path to save checkpoint and result embeding
parser.add_argument('--num_iterations', type=int, default=60001)
parser.add_argument('--num_input_channels', type=int, default=360) #features' dimention, mesh-360, toydadta-39
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--num_points', type=int, default=1000) # number of points in each mesh or point cloud

FLAGS = parser.parse_args()

ARCHITECTURE = FLAGS.architecture
DATASET_PATH = FLAGS.dataset_path
RESULTS_PATH = FLAGS.results_path
NUM_ITERATIONS = FLAGS.num_iterations
LEARNING_RATE = FLAGS.learning_rate
NUM_POINTS = FLAGS.num_points
NUM_INPUT_CHANNEL = FLAGS.num_input_channels
CONTINUE_TRAIN = 0
IS_TRAIN = 1
IS_TRAIN_BIG = 1 # 0 means fine-tune stage
FAUST_or_SCAPE = 1 # specify the dataset, 1-faust, 0-scape
BASE_num = 16


MODEL_TYPE = 0 # 0-proposed SBL

if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

if MODEL_TYPE == 0:
    gcn_adj1 = tf.sparse_placeholder(tf.float32, shape=[NUM_POINTS,NUM_POINTS])
    lpmtx1 = tf.sparse_placeholder(tf.float32, shape=[NUM_POINTS,NUM_POINTS])
    fea1 = tf.placeholder(tf.float32, shape=[NUM_POINTS, NUM_INPUT_CHANNEL])



#*********************************************models_and_loss************************************************

if MODEL_TYPE == 0:

    rebase1,rebases1 = get_model_baselearning(fea1, gcn_adj1, re_use=False, is_train_big=IS_TRAIN_BIG)

    orthcal1 = tf.matmul(rebase1, rebase1, transpose_a=True)
    loss0 = tf.losses.mean_squared_error(tf.eye(NUM_POINTS), orthcal1)

    lamda1_0 = tf.sparse_tensor_dense_matmul(lpmtx1,rebase1)
    lamda1_1 = tf.matmul(rebase1, lamda1_0, transpose_a=True)
    loss_matrix = tf.abs(lamda1_1 * (1.0 - tf.eye(NUM_POINTS)))
    loss1_1 = tf.reduce_mean(loss_matrix)

    lamda1_2 = tf.nn.l2_normalize(tf.sparse_tensor_dense_matmul(gcn_adj1, rebase1), axis=0)
    lamda1_3 = tf.reduce_sum(lamda1_2 * tf.nn.l2_normalize(rebase1, axis=0), axis=0)
    loss1_3 = 1.0 - tf.reduce_mean(lamda1_3)

    loss2 = tf.losses.mean_squared_error(tf.eye(BASE_num), tf.matmul(rebases1, rebases1, transpose_a=True))

    lamda3_1 = tf.nn.l2_normalize(tf.sparse_tensor_dense_matmul(gcn_adj1, rebases1), axis=0)
    lamda3_2 = tf.reduce_sum(lamda3_1 * tf.nn.l2_normalize(rebases1, axis=0),axis=0)
    loss3_1 = 1.0 - tf.reduce_mean(lamda3_2)

    lamda2_1 = tf.matmul(rebases1, tf.sparse_tensor_dense_matmul(lpmtx1, rebases1), transpose_a=True)
    loss_matrix2 = lamda2_1 * (1.0 - tf.eye(BASE_num))
    loss2_1 = tf.reduce_mean(tf.abs(loss_matrix2))

    if IS_TRAIN_BIG == 1:
        myloss = 1 * loss0 + 5 * loss1_1 + 5 * loss1_3 + 1 * loss2 + 5 * loss3_1 + 5 * loss2_1#
    else:
        myloss = 0.2 * loss2 + 1 * loss3_1 + 1 * loss2_1

batch = tf.Variable(0, trainable=False)

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(myloss, global_step=batch)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

ckpt = tf.train.get_checkpoint_state(os.path.dirname(RESULTS_PATH))
if ckpt and ckpt.model_checkpoint_path and CONTINUE_TRAIN:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Checkpoint restored")

#*******************************************load_data****************************************************
if IS_TRAIN == 1 and FAUST_or_SCAPE == 1:

    adj_path = []
    for root, _, fnames in sorted(os.walk(DATASET_PATH+'train_adj')):
        for fname in fnames:
            path = os.path.join(root, fname)
            adj_path.append(path)

    fea_path = []
    for root, _, fnames in sorted(os.walk(DATASET_PATH+'train_feature')):
        for fname in fnames:
            path = os.path.join(root, fname)
            fea_path.append(path)



if IS_TRAIN == 0 and FAUST_or_SCAPE == 1:

    test_fea_path = []
    for root, _, fnames in sorted(os.walk(DATASET_PATH+'test_feature')):
        for fname in fnames:
            path = os.path.join(root, fname)
            test_fea_path.append(path)

    test_adj_path = []
    for root, _, fnames in sorted(os.walk(DATASET_PATH+'test_adj')):
        for fname in fnames:
            path = os.path.join(root, fname)
            test_adj_path.append(path)


def get_input(iter):
    # read training data
    ind = iter

    tempname = adj_path[ind]
    temp = sio.loadmat(tempname)
    lpmtx_input = temp['normlap']
    lpmtx_input = lpmtx_input.astype(np.float32)
    lpmtx_input = scipy.sparse.coo_matrix(lpmtx_input)

    sparse_adj_input = temp['graph_adj']
    sparse_adj_input = sparse_adj_input.astype(np.float32)
    sparse_adj_input = scipy.sparse.coo_matrix(sparse_adj_input)


    tempname = fea_path[ind]
    temp = sio.loadmat(tempname)
    hand_feainput = temp['fea']

    # centralization of the coordinate
    hand_feainput[:, 0:1] = (hand_feainput[:, 0:1] - np.mean(hand_feainput[:, 0:1]))
    hand_feainput[:, 1:2] = (hand_feainput[:, 1:2] - np.mean(hand_feainput[:, 1:2]))
    hand_feainput[:, 2:3] = (hand_feainput[:, 2:3] - np.min(hand_feainput[:, 2:3]))

    return sparse_adj_input, lpmtx_input, hand_feainput


def test_input(ind):
    # read testing data
    tempname = test_adj_path[ind]
    temp = sio.loadmat(tempname)
    lpmtx_input = temp['normlap']
    lpmtx_input = lpmtx_input.astype(np.float32)
    lpmtx_input = scipy.sparse.coo_matrix(lpmtx_input)

    sparse_adj_input = temp['graph_adj']
    sparse_adj_input = sparse_adj_input.astype(np.float32)
    sparse_adj_input = scipy.sparse.coo_matrix(sparse_adj_input)

    tempname = test_fea_path[ind]

    temp = sio.loadmat(tempname)
    hand_feainput = temp['fea']
    hand_feainput[:, 0:1] = (hand_feainput[:, 0:1] - np.mean(hand_feainput[:, 0:1]))
    hand_feainput[:, 1:2] = (hand_feainput[:, 1:2] - np.mean(hand_feainput[:, 1:2]))
    hand_feainput[:, 2:3] = (hand_feainput[:, 2:3] - np.min(hand_feainput[:, 2:3]))

    return sparse_adj_input, lpmtx_input, hand_feainput


for iter in range(NUM_ITERATIONS):
    if IS_TRAIN == 1:

        if FAUST_or_SCAPE == 1:
            train_num = 1000

        ind1 = random.randint(0, train_num)

        if MODEL_TYPE == 0:
            sparse_adj1, lpmtx_input1, fea_input1 = get_input(ind1)
            sparse_adj1_ind = np.concatenate(
                [np.expand_dims(sparse_adj1.row, axis=1), np.expand_dims(sparse_adj1.col, axis=1)], axis=1)
            sparse_adj1_val = sparse_adj1.data
            sparse_adj1_shp = sparse_adj1.shape
            lpmtx_input1_ind = np.concatenate(
                [np.expand_dims(lpmtx_input1.row, axis=1), np.expand_dims(lpmtx_input1.col, axis=1)], axis=1)
            lpmtx_input1_val = lpmtx_input1.data
            lpmtx_input1_shp = lpmtx_input1.shape

        if iter % 1000 == 0:
            if MODEL_TYPE == 0:
                tempsee = myloss.eval(feed_dict={gcn_adj1: (sparse_adj1_ind, sparse_adj1_val, sparse_adj1_shp),
                                                 lpmtx1: (lpmtx_input1_ind, lpmtx_input1_val, lpmtx_input1_shp),
                                                 fea1: fea_input1})

                print("Iteration %d, Index %d, loss %g" % (iter, ind1, tempsee))

        if MODEL_TYPE == 0:
            train_step.run(feed_dict={gcn_adj1: (sparse_adj1_ind, sparse_adj1_val, sparse_adj1_shp),
                                      lpmtx1: (lpmtx_input1_ind, lpmtx_input1_val, lpmtx_input1_shp),
                                      fea1: fea_input1})


    elif IS_TRAIN == 0:


        if FAUST_or_SCAPE == 0:
            temptestnum = 69
            for testi in range(0 + 0, temptestnum + 0):

                if MODEL_TYPE == 0 or MODEL_TYPE == 3 or MODEL_TYPE == 1:
                    sparse_adj1, lpmtx_input1, fea_input1 = test_input(testi)

                    sparse_adj1_ind = np.concatenate(
                        [np.expand_dims(sparse_adj1.row, axis=1), np.expand_dims(sparse_adj1.col, axis=1)], axis=1)
                    sparse_adj1_val = sparse_adj1.data
                    sparse_adj1_shp = sparse_adj1.shape
                    lpmtx_input1_ind = np.concatenate(
                        [np.expand_dims(lpmtx_input1.row, axis=1), np.expand_dims(lpmtx_input1.col, axis=1)], axis=1)
                    lpmtx_input1_val = lpmtx_input1.data
                    lpmtx_input1_shp = lpmtx_input1.shape

                    res_base = rebase1.eval(
                        feed_dict={fea1: fea_input1, gcn_adj1: (sparse_adj1_ind, sparse_adj1_val, sparse_adj1_shp),
                                   lpmtx1: (lpmtx_input1_ind, lpmtx_input1_val, lpmtx_input1_shp)})
                    res_bases = rebases1.eval(
                        feed_dict={fea1: fea_input1, gcn_adj1: (sparse_adj1_ind, sparse_adj1_val, sparse_adj1_shp),
                                   lpmtx1: (lpmtx_input1_ind, lpmtx_input1_val, lpmtx_input1_shp)})

                    sio.savemat(RESULTS_PATH + test_adj_path[testi][24:60],#24 means the start position of files' name
                                {'re_base': res_base, 're_bases': res_bases})


        elif FAUST_or_SCAPE == 1:
            temptestnum = 182

            for testi in range(0 + 0, temptestnum + 0):

                if MODEL_TYPE == 0:
                    sparse_adj1, lpmtx_input1, fea_input1 = test_input(testi)
                    sparse_adj1_ind = np.concatenate(
                        [np.expand_dims(sparse_adj1.row, axis=1), np.expand_dims(sparse_adj1.col, axis=1)], axis=1)
                    sparse_adj1_val = sparse_adj1.data
                    sparse_adj1_shp = sparse_adj1.shape
                    lpmtx_input1_ind = np.concatenate(
                        [np.expand_dims(lpmtx_input1.row, axis=1), np.expand_dims(lpmtx_input1.col, axis=1)], axis=1)
                    lpmtx_input1_val = lpmtx_input1.data
                    lpmtx_input1_shp = lpmtx_input1.shape

                    res_base = rebase1.eval(
                        feed_dict={fea1: fea_input1, gcn_adj1: (sparse_adj1_ind, sparse_adj1_val, sparse_adj1_shp),
                                   lpmtx1: (lpmtx_input1_ind, lpmtx_input1_val, lpmtx_input1_shp)})
                    res_bases = rebases1.eval(
                        feed_dict={fea1: fea_input1, gcn_adj1: (sparse_adj1_ind, sparse_adj1_val, sparse_adj1_shp),
                                   lpmtx1: (lpmtx_input1_ind, lpmtx_input1_val, lpmtx_input1_shp)})

                    sio.savemat(RESULTS_PATH + test_adj_path[testi][19:60],#19 means the start position of files' name
                                {'re_base': res_base, 're_bases': res_bases})

            print("test_index %d " % testi)

    if iter % 10000 == 0 and iter > 1:
        saver.save(sess, RESULTS_PATH + 'model.ckpt', global_step=iter)
        print("checkpoint saved")

