from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import tensorflow as tf
import argparse
import scannet_dataset

import tf_utils.provider as provider
import models.pointSIFT_pointnet as SEG_MODEL

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=1000, help='epoch to run[default: 1000]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size during training[default: 32')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate[default: 1e-3]')
parser.add_argument('--save_path', default='model_param', help='model param path')
parser.add_argument('--data_path', default='data', help='scannet dataset path')
parser.add_argument('--train_log_path', default='log/pointSIFT_train')
parser.add_argument('--test_log_path', default='log/pointSIFT_test')
parser.add_argument('--gpu_num', type=int, default=1, help='number of GPU to train')

# basic params..

FLAGS = parser.parse_args()
BATCH_SZ = FLAGS.batch_size
LEARNING_RATE = FLAGS.learning_rate
MAX_EPOCH = FLAGS.max_epoch
SAVE_PATH = FLAGS.save_path
DATA_PATH = FLAGS.data_path
TRAIN_LOG_PATH = FLAGS.train_log_path
TEST_LOG_PATH = FLAGS.test_log_path
GPU_NUM = FLAGS.gpu_num
BATCH_PER_GPU = BATCH_SZ // GPU_NUM

NUM_CLASS = 21

# lr params..
DECAY_STEP = 200000
DECAY_RATE = 0.7

# bn params..
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

class SegTrainer(object):
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.train_sz = 0
        self.test_sz = 0
        self.point_sz = 8192

        # batch loader init....
        self.batch_loader = None
        self.batch_sz = BATCH_SZ

        # net param...
        self.point_pl = None
        self.label_pl = None
        self.smpws_pl = None
        self.is_train_pl = None
        self.ave_tp_pl = None
        self.net = None
        self.end_point = None
        self.bn_decay = None
        self.loss = None
        self.optimizer = None
        self.train_op = None
        self.predict = None
        self.TP = None
        self.batch = None  # record the training step..

        # summary
        self.ave_tp_summary = None

        # list for multi gpu tower..
        self.tower_grads = []
        self.net_gpu = []
        self.total_loss_gpu_list = []

    def load_data(self):
        assert os.path.exists(DATA_PATH), 'train_data not found !!!'
        self.train_data = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=self.point_sz, split='train')
        self.test_data = scannet_dataset.ScannetDatasetWholeScene(root=DATA_PATH, npoints=self.point_sz, split='test')
        self.train_sz = self.train_data.__len__()
        self.test_sz = self.test_data.__len__()
        print('train size %d and test size %d' % (self.train_sz, self.test_sz))

    def get_learning_rate(self):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE,
                                                   self.batch * BATCH_SZ,
                                                   DECAY_STEP,
                                                   DECAY_RATE,
                                                   staircase=True)
        learning_rate = tf.maximum(learning_rate, 1e-5)
        tf.summary.scalar('learning rate', learning_rate)
        return learning_rate

    def get_bn_decay(self):
        bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,
                                                 self.batch * BATCH_SZ,
                                                 BN_DECAY_DECAY_STEP,
                                                 BN_DECAY_DECAY_RATE,
                                                 staircase=True)
        bn_momentum = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        tf.summary.scalar('bn_decay', bn_momentum)
        return bn_momentum

    def get_batch_wdp(self, dataset, idxs, start_idx, end_idx):
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.point_sz, 3))
        batch_label = np.zeros((bsize, self.point_sz), dtype=np.int32)
        batch_smpw = np.zeros((bsize, self.point_sz), dtype=np.float32)
        for i in range(bsize):
            ps, seg, smpw = dataset[idxs[i + start_idx]]
            batch_data[i, ...] = ps
            batch_label[i, :] = seg
            batch_smpw[i, :] = smpw

            dropout_ratio = np.random.random() * 0.875  # 0-0.875
            drop_idx = np.where(np.random.random((ps.shape[0])) <= dropout_ratio)[0]

            batch_data[i, drop_idx, :] = batch_data[i, 0, :]
            batch_label[i, drop_idx] = batch_label[i, 0]
            batch_smpw[i, drop_idx] *= 0
        return batch_data, batch_label, batch_smpw

    def get_batch(self, dataset, idxs, start_idx, end_idx):
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.point_sz, 3))
        batch_label = np.zeros((bsize, self.point_sz), dtype=np.int32)
        batch_smpw = np.zeros((bsize, self.point_sz), dtype=np.float32)
        for i in range(bsize):
            ps, seg, smpw = dataset[idxs[i + start_idx]]
            batch_data[i, ...] = ps
            batch_label[i, :] = seg
            batch_smpw[i, :] = smpw
        return batch_data, batch_label, batch_smpw

    @staticmethod
    def ave_gradient(tower_grad):
        ave_gradient = []
        for gpu_data in zip(*tower_grad):
            grads = []
            for g, k in gpu_data:
                t_g = tf.expand_dims(g, axis=0)
                grads.append(t_g)
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)
            key = gpu_data[0][1]
            ave_gradient.append((grad, key))
        return ave_gradient

    # cpu part of graph
    def build_g_cpu(self):
        self.batch = tf.Variable(0, name='batch', trainable=False)
        self.point_pl, self.label_pl, self.smpws_pl = SEG_MODEL.placeholder_inputs(self.batch_sz, self.point_sz)
        self.is_train_pl = tf.placeholder(dtype=tf.bool, shape=())
        self.ave_tp_pl = tf.placeholder(dtype=tf.float32, shape=())
        self.optimizer = tf.train.AdamOptimizer(self.get_learning_rate())
        self.bn_decay = self.get_bn_decay()

        SEG_MODEL.get_model(self.point_pl, self.is_train_pl, num_class=NUM_CLASS, bn_decay=self.bn_decay)

    # graph for each gpu, reuse params...
    def build_g_gpu(self, gpu_idx):
        print("build graph in gpu %d" % gpu_idx)
        with tf.device('/gpu:%d' % gpu_idx), tf.name_scope('gpu_%d' % gpu_idx) as scope:
            point_cloud_slice = tf.slice(self.point_pl, [gpu_idx * BATCH_PER_GPU, 0, 0], [BATCH_PER_GPU, -1, -1])
            label_slice = tf.slice(self.label_pl, [gpu_idx * BATCH_PER_GPU, 0], [BATCH_PER_GPU, -1])
            smpws_slice = tf.slice(self.smpws_pl, [gpu_idx * BATCH_PER_GPU, 0], [BATCH_PER_GPU, -1])
            net, end_point = SEG_MODEL.get_model(point_cloud_slice, self.is_train_pl, num_class=NUM_CLASS,
                                                 bn_decay=self.bn_decay)
            SEG_MODEL.get_loss(net, label_slice, smpw=smpws_slice)
            loss = tf.get_collection('losses', scope=scope)
            total_loss = tf.add_n(loss, name='total_loss')
            for _i in loss + [total_loss]:
                tf.summary.scalar(_i.op.name, _i)

            gvs = self.optimizer.compute_gradients(total_loss)
            self.tower_grads.append(gvs)
            self.net_gpu.append(net)
            self.total_loss_gpu_list.append(total_loss)

    def build_graph(self):
        with tf.device('/cpu:0'):
            self.build_g_cpu()
            self.tower_grads = []
            self.net_gpu = []
            self.total_loss_gpu_list = []

            for i in range(GPU_NUM):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    self.build_g_gpu(i)

            self.net = tf.concat(self.net_gpu, axis=0)
            self.loss = tf.reduce_mean(self.total_loss_gpu_list)

            # get training op
            gvs = self.ave_gradient(self.tower_grads)
            self.train_op = self.optimizer.apply_gradients(gvs, global_step=self.batch)
            self.predict = tf.cast(tf.argmax(self.net, axis=2), tf.int32)
            self.TP = tf.reduce_sum(
                tf.cast(tf.equal(self.predict, self.label_pl), tf.float32)) / self.batch_sz / self.point_sz
            tf.summary.scalar('TP', self.TP)
            tf.summary.scalar('total_loss', self.loss)

    def training(self):
        with tf.Graph().as_default():
            self.build_graph()
            # merge operator (for tensorboard)
            merged = tf.summary.merge_all()
            iter_in_epoch = self.train_sz // self.batch_sz
            saver = tf.train.Saver()

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            best_acc = 0.0
            with tf.Session(config=config) as sess:
                train_writer = tf.summary.FileWriter(TRAIN_LOG_PATH, sess.graph)
                evaluate_writer = tf.summary.FileWriter(TEST_LOG_PATH, sess.graph)
                sess.run(tf.global_variables_initializer())
                epoch_sz = MAX_EPOCH
                tic = time.time()
                for epoch in range(epoch_sz):
                    ave_loss = 0
                    train_idxs = np.arange(0, self.train_data.__len__())
                    np.random.shuffle(train_idxs)
                    for _iter in range(iter_in_epoch):
                        start_idx = _iter * self.batch_sz
                        end_idx = (_iter + 1) * self.batch_sz
                        batch_data, batch_label, batch_smpw = self.get_batch_wdp(self.train_data, train_idxs,
                                                                                 start_idx, end_idx)
                        aug_data = provider.rotate_point_cloud_z(batch_data)
                        loss, _, summary, step = sess.run([self.loss, self.train_op, merged, self.batch],
                                                          feed_dict={self.point_pl: aug_data,
                                                                     self.label_pl: batch_label,
                                                                     self.smpws_pl: batch_smpw,
                                                                     self.is_train_pl: True})
                        ave_loss += loss
                        train_writer.add_summary(summary, step)
                    ave_loss /= iter_in_epoch
                    print("epoch %d , loss is %f take %.3f s" % (epoch + 1, ave_loss, time.time() - tic))
                    tic = time.time()
                    if (epoch + 1) % 5 == 0:
                        acc = self.evaluate_one_epoch(sess, evaluate_writer, step, epoch)
                        if acc > best_acc:
                            _path = saver.save(sess, os.path.join(SAVE_PATH, "best_seg_model_%d.ckpt" % (epoch + 1)))
                            print("epoch %d, best saved in file: " % (epoch + 1), _path)
                            best_acc = acc
                _path = saver.save(sess, os.path.join(SAVE_PATH, 'train_base_seg_model.ckpt'))
                print("Model saved in file: ", _path)

    def evaluate_one_epoch(self, sess, test_writer, step, epoch):
        is_training = False
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(NUM_CLASS)]
        total_correct_class = [0 for _ in range(NUM_CLASS)]

        total_correct_vox = 0
        total_seen_vox = 0
        total_seen_class_vox = [0 for _ in range(NUM_CLASS)]
        total_correct_class_vox = [0 for _ in range(NUM_CLASS)]

        labelweights_vox = np.zeros(21)
        is_continue_batch = False

        extra_batch_data = np.zeros((0, self.point_sz, 3))
        extra_batch_label = np.zeros((0, self.point_sz))
        extra_batch_smpw = np.zeros((0, self.point_sz))
        batch_data, batch_label, batch_smpw = None, None, None
        print("---EVALUATE %d EPOCH---" % (epoch + 1))
        for batch_idx in range(self.test_data.__len__()):
            if not is_continue_batch:
                batch_data, batch_label, batch_smpw = self.test_data[batch_idx]
                batch_data = np.concatenate((batch_data, extra_batch_data), axis=0)
                batch_label = np.concatenate((batch_label, extra_batch_label), axis=0)
                batch_smpw = np.concatenate((batch_smpw, extra_batch_smpw), axis=0)
            else:
                batch_data_tmp, batch_label_tmp, batch_smpw_tmp = self.test_data[batch_idx]
                batch_data = np.concatenate((batch_data, batch_data_tmp), axis=0)
                batch_label = np.concatenate((batch_label, batch_label_tmp), axis=0)
                batch_smpw = np.concatenate((batch_smpw, batch_smpw_tmp), axis=0)
            if batch_data.shape[0] < self.batch_sz:
                is_continue_batch = True
                continue
            while batch_data.shape[0] >= self.batch_sz:
                is_continue_batch = False
                if batch_data.shape[0] == self.batch_sz:
                    extra_batch_data = np.zeros((0, self.point_sz, 3))
                    extra_batch_label = np.zeros((0, self.point_sz))
                    extra_batch_smpw = np.zeros((0, self.point_sz))
                else:
                    extra_batch_data = batch_data[self.batch_sz:, :, :]
                    extra_batch_label = batch_label[self.batch_sz:, :]
                    extra_batch_smpw = batch_smpw[self.batch_sz:, :]
                    batch_data = batch_data[: self.batch_sz, :, :]
                    batch_label = batch_label[: self.batch_sz, :]
                    batch_smpw = batch_smpw[: self.batch_sz, :]
                aug_data = batch_data
                net, loss_val = sess.run([self.net, self.loss], feed_dict={self.point_pl: aug_data,
                                                                           self.label_pl: batch_label,
                                                                           self.smpws_pl: batch_smpw,
                                                                           self.is_train_pl: is_training})
                pred_val = np.argmax(net, axis=2)
                correct = np.sum((pred_val == batch_label) & (batch_label > 0) & (batch_smpw > 0))
                total_correct += correct
                total_seen += np.sum((batch_label > 0) & (batch_smpw > 0))
                loss_sum += loss_val

                for l in range(NUM_CLASS):
                    total_seen_class[l] += np.sum((batch_label == l) & (batch_smpw > 0))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) & (batch_smpw > 0))
                for b in range(batch_label.shape[0]):
                    uvlabel = provider.point_cloud_label_to_surface_voxel_label(aug_data[b, batch_smpw[b, :] > 0, :],
                                                                                np.concatenate((np.expand_dims(batch_label[b, batch_smpw[b, :] > 0], 1),
                                                                                                np.expand_dims(pred_val[b, batch_smpw[b, :] > 0], 1)), axis=1),
                                                                                res=0.02)
                    total_correct_vox += np.sum((uvlabel[:, 0] == uvlabel[:, 1]) & (uvlabel[:, 0] > 0))
                    total_seen_vox += np.sum(uvlabel[:, 0] > 0)
                    tmp, _ = np.histogram(uvlabel[:, 0], range(22))
                    labelweights_vox += tmp
                    for l in range(NUM_CLASS):
                        total_seen_class_vox[l] += np.sum(uvlabel[:, 0] == l)
                        total_correct_class_vox[l] += np.sum((uvlabel[:, 0] == l) & (uvlabel[:, 1] == l))
                batch_data = extra_batch_data
                batch_label = extra_batch_label
                batch_smpw = extra_batch_smpw

        print('eval whole scene mean loss: %f' % (loss_sum / float(self.test_sz)))
        print('eval whole scene point accuracy vox: %f' % (total_correct_vox / float(total_seen_vox)))
        print('eval whole scene point avg class acc vox: %f' % (np.mean(np.array(total_correct_class_vox[1:]) / (np.array(total_seen_class_vox[1:], dtype=np.float) + 1e-6))))
        print('eval whole scene point accuracy: %f' % (total_correct / float(total_seen)))
        print('eval whole scene point avg class acc: %f' % (np.mean(np.array(total_correct_class[1:]) / (np.array(total_seen_class[1:], dtype=np.float) + 1e-6))))
        labelweights_vox = labelweights_vox[1:].astype(np.float32) / np.sum(labelweights_vox[1:].astype(np.float32))
        caliweights = np.array(
            [0.388, 0.357, 0.038, 0.033, 0.017, 0.022, 0.016, 0.025, 0.002, 0.002, 0.002, 0.007, 0.006, 0.022, 0.004,
             0.0004, 0.003, 0.002, 0.024, 0.029])
        caliacc = np.average(
            np.array(total_correct_class_vox[1:]) / (np.array(total_seen_class_vox[1:], dtype=np.float) + 1e-6),
            weights=caliweights)
        print('eval whole scene point calibrated average acc vox: %f' % caliacc)

        per_class_str = 'vox based --------'
        for l in range(1, NUM_CLASS):
            per_class_str += 'class %d weight: %f, acc: %f; ' % (l, labelweights_vox[l - 1], total_correct_class_vox[l] / float(total_seen_class_vox[l]))
        print(per_class_str)
        print("caliacc is %f " % caliacc)
        ave_tp_summary = tf.summary.Summary(value=[tf.summary.Summary.Value(tag="TP", simple_value=caliacc)])
        test_writer.add_summary(ave_tp_summary, step)
        return caliacc


if __name__ == '__main__':
    trainer = SegTrainer()
    trainer.load_data()
    trainer.training()
