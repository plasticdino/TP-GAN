from __future__ import division
from __future__ import print_function
import os, sys
import time
import datetime

import tensorflow as tf

from six.moves import xrange

from models.ops import *
from scripts.utils import *

from scripts.create_dataset import *
from time import localtime, strftime
import random
import pickle

relu = tf.nn.relu

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return concat(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


class DCGAN(object):
    def __init__(self, sess, get_checkpoint, params, testing):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.get_checkpoint = get_checkpoint
        self.sample_run_num = 15
        self.testing = testing
        # batch normalization : deals with poor initialization helps gradient flow
        random.seed()
        self.DeepFacePath = 'models/DeepFace168.pickle'
        self.loadDeepFace(self.DeepFacePath)
        self.build_model(params)

        self.data = MultiPIE(params, LOAD_60_LABEL=params.LOAD_60_LABEL,
                             RANDOM_VERIFY=params.RANDOM_VERIFY,
                             MIRROR_TO_ONE_SIDE=True,
                             testing=self.testing)

    def build_model(self, params):

        # hold all four
        # Note: not true, if params.WITHOUT_CODEMAP is true, then here is pure images without codemap and 3 channels
        # mirror concatenate
        mc = lambda left: concat([left, left[:, :, ::-1, :]], 3)
        self.images_with_code = tf.placeholder(tf.float32,
                                               [params.batch_size] + [params.output_size, params.output_size, params.CHANNEL],
                                               name='images_with_code')
        self.sample_images = tf.placeholder(tf.float32,
                                            [params.batch_size] + [params.output_size, params.output_size,
                                                                   params.CHANNEL],
                                            name='sample_images')
        # self.docker_input = tf.placeholder(tf.float32,1, name = 'test_docker')
        # self.docker_output = self.docker_input + 1

        if params.WITHOUT_CODEMAP:
            self.images = self.images_with_code
            self.sample_images_nocode = self.sample_images
        else:
            self.images = tf.split(3, 2, self.images_with_code)[0]
            self.sample_images_nocode = tf.split(3, 2, self.sample_images)[0]

        self.g_images = self.images  # tf.reduce_mean(self.images, axis=3, keep_dims=True)
        self.g_samples = self.sample_images_nocode  # tf.reduce_mean(self.sample_images_nocode, axis=3, keep_dims=True)

        self.g32_images_with_code = tf.image.resize_bilinear(self.images_with_code, [32, 32])
        self.g64_images_with_code = tf.image.resize_bilinear(self.images_with_code, [64, 64])

        self.g32_sampleimages_with_code = tf.image.resize_bilinear(self.sample_images, [32, 32])
        self.g64_sampleimages_with_code = tf.image.resize_bilinear(self.sample_images, [64, 64])

        self.labels = tf.placeholder(tf.float32, [params.batch_size] + [params.output_size, params.output_size, 3],
                                     name='label_images')
        self.poselabels = tf.placeholder(tf.int32, [params.batch_size])
        self.idenlabels = tf.placeholder(tf.int32, [params.batch_size])
        self.landmarklabels = tf.placeholder(tf.float32, [params.batch_size, 5 * 2])
        self.g_labels = self.labels  # tf.reduce_mean(self.labels, 3, keep_dims=True)
        self.g8_labels = tf.image.resize_bilinear(self.g_labels, [8, 8])
        self.g16_labels = tf.image.resize_bilinear(self.g_labels, [16, 16])
        self.g32_labels = tf.image.resize_bilinear(self.g_labels, [32, 32])
        self.g64_labels = tf.image.resize_bilinear(self.g_labels, [64, 64])

        self.eyel = tf.placeholder(tf.float32, [params.batch_size, EYE_H, EYE_W, 3],name='eyel')
        self.eyer = tf.placeholder(tf.float32, [params.batch_size, EYE_H, EYE_W, 3],name='eyer')
        self.nose = tf.placeholder(tf.float32, [params.batch_size, NOSE_H, NOSE_W, 3],name='nose')
        self.mouth = tf.placeholder(tf.float32, [params.batch_size, MOUTH_H, MOUTH_W, 3],name='mouth')

        self.eyel_label = tf.placeholder(tf.float32, [params.batch_size, EYE_H, EYE_W, 3], name='eyel_label')
        self.eyer_label = tf.placeholder(tf.float32, [params.batch_size, EYE_H, EYE_W, 3], name='eyer_label')
        self.nose_label = tf.placeholder(tf.float32, [params.batch_size, NOSE_H, NOSE_W, 3],name='nose_label')
        self.mouth_label = tf.placeholder(tf.float32, [params.batch_size, MOUTH_H, MOUTH_W, 3],name='mouth_label')

        self.eyel_sam = tf.placeholder(tf.float32, [params.batch_size, EYE_H, EYE_W, 3],name='eyel_sam')
        self.eyer_sam = tf.placeholder(tf.float32, [params.batch_size, EYE_H, EYE_W, 3],name='eyer_sam')
        self.nose_sam = tf.placeholder(tf.float32, [params.batch_size, NOSE_H, NOSE_W, 3],name='nose_sam')
        self.mouth_sam = tf.placeholder(tf.float32, [params.batch_size, MOUTH_H, MOUTH_W, 3],name='mouth_sam')

        # feats contains: self.feat128, self.feat64, self.feat32, self.feat16, self.feat8, self.feat
        self.G_eyel, self.c_eyel = self.partRotator(self.eyel, "PartRotator_eyel", params)
        self.G_eyer, self.c_eyer = self.partRotator(concat([self.eyer, self.eyel], axis=3), "PartRotator_eyer", params)
        self.G_nose, self.c_nose = self.partRotator(self.nose, "PartRotator_nose", params)
        self.G_mouth, self.c_mouth = self.partRotator(self.mouth, "PartRotator_mouth", params)

        self.G_eyel_sam, self.c_eyel_sam = self.partRotator(self.eyel_sam, "PartRotator_eyel", params, reuse=True)
        self.G_eyer_sam, self.c_eyer_sam = self.partRotator(concat([self.eyer_sam, self.eyel_sam], axis=3),
                                                            "PartRotator_eyer", params, reuse=True)
        self.G_nose_sam, self.c_nose_sam = self.partRotator(self.nose_sam, "PartRotator_nose", params, reuse=True)
        self.G_mouth_sam, self.c_mouth_sam = self.partRotator(self.mouth_sam, "PartRotator_mouth", params, reuse=True)

        self.z = tf.random_normal([params.batch_size, params.z_dim], mean=0.0, stddev=0.02, seed=2017)

        # tf.placeholder(tf.float32, [params.batch_size, params.z_dim], name='z')

        self.feats = self.generator(mc(self.images_with_code), params.batch_size, params, name="encoder")
        self.feats += (mc(self.images_with_code), mc(self.g64_images_with_code), mc(self.g32_images_with_code),
                       self.G_eyel, self.G_eyer, self.G_nose, self.G_mouth,
                       self.c_eyel, self.c_eyer, self.c_nose, self.c_mouth,)
        self.check_sel128, self.check_sel64, self.check_sel32, self.check_sel16, self.check_sel8, self.G, self.G2, self.G3 = \
            self.decoder(params, *self.feats, batch_size=params.batch_size)
        self.poselogits, self.identitylogits, self.Glandmark = self.FeaturePredict(self.feats[5])

        sample_feats = self.generator(mc(self.sample_images), params.batch_size, params, name="encoder", reuse=True)
        self.sample512 = sample_feats[-1]
        sample_feats += (
            mc(self.sample_images), mc(self.g64_sampleimages_with_code), mc(self.g32_sampleimages_with_code),
            self.G_eyel_sam, self.G_eyer_sam, self.G_nose_sam, self.G_mouth_sam,
            self.c_eyel_sam, self.c_eyer_sam, self.c_nose_sam, self.c_mouth_sam,)
        self.sample_generator = self.decoder(params, *sample_feats, batch_size=params.batch_size, reuse=True)
        if not params.DF:  # df = true
            self.D, self.D_logits = self.discriminator(self.g_labels)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        else:
            print("Using local discriminator!")
            self.D, self.D_logits = self.discriminatorLocal(params, self.g_labels)
            self.D_, self.D_logits_ = self.discriminatorLocal(params, self.G, reuse=True)

        if 'f' in params.MODE:
            # self.verify_images_masked = tf.mul(self.verify_images, self.masks_binary)
            # can not apply mask !!!
            # self.Dv, self.Dv_logits = self.discriminatorVerify(self.labels, self.verify_images)
            _, _, _, _, self.G_pool5, self.Gvector = self.FeatureExtractDeepFace(
                tf.reduce_mean(self.G, axis=3, keep_dims=True))
            _, _, _, _, self.label_pool5, self.labelvector = self.FeatureExtractDeepFace(
                tf.reduce_mean(self.g_labels, axis=3, keep_dims=True), reuse=True)
            _, _, _, _, _, self.samplevector = self.FeatureExtractDeepFace(
                tf.reduce_mean(self.sample_images_nocode, axis=3, keep_dims=True), reuse=True)
            # self.Dv, self.Dv_logits = self.discriminatorClassify(self.Gvector)
            # self.dv_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(self.Dv_logits, self.verify_labels))
            self.dv_loss = tf.reduce_mean(tf.abs(self.Gvector - self.labelvector))
            self.dv_loss += tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.abs(self.G_pool5 - self.label_pool5), 1), 1))
            # self.dv_sum = histogram_summary("dv_", self.Dv)
        else:
            self.dv_loss = 0
        # self.d__sum = histogram_summary("d_", self.D_)
        # self.d_sum = histogram_summary("d", self.D)
        # self.G_sum = image_summary("G", self.G)

        # basic loss

        # self.d_loss_real = tf.reduce_mean(self.D_logits)
        # self.d_loss_fake = -tf.reduce_mean(self.D_logits_)
        # self.g_loss_adver = -tf.reduce_mean(self.D_logits_)
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D) * 0.9))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss_adver = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_) * 0.9))

        # self.mark_regression_loss = tf.reduce_mean(tf.square(tf.abs(self.landmarklabels-self.Glandmark)))
        # self.poseloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.poselogits, self.poselabels))
        self.idenloss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.identitylogits, labels=self.idenlabels))

        self.eyel_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.abs(self.c_eyel - self.eyel_label), 1), 1))
        self.eyer_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.abs(self.c_eyer - self.eyer_label), 1), 1))
        self.nose_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.abs(self.c_nose - self.nose_label), 1), 1))
        self.mouth_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.abs(self.c_mouth - self.mouth_label), 1), 1))
        # rotation L1 / L2 loss in g_loss
        # one8 = tf.ones([1,8,4,1],tf.float32)
        # mask8 = concat([one8, one8], 2)
        # mask16 = tf.image.resize_nearest_neighbor(mask8, size=[16, 16])
        # mask32 = tf.image.resize_nearest_neighbor(mask8, size=[32, 32])
        # mask64 = tf.image.resize_nearest_neighbor(mask8, size=[64, 64])
        # mask128 = tf.image.resize_nearest_neighbor(mask8, size=[128, 128])
        # use L2 for 128, L1 for others. mask emphasize left side.
        errL1 = tf.abs(self.G - self.g_labels)  # * mask128
        errL1_2 = tf.abs(self.G2 - self.g64_labels)  # * mask64
        errL1_3 = tf.abs(self.G3 - self.g32_labels)  # * mask32
        # errcheck8 = tf.abs(self.check_sel8 - self.g8_labels) #* mask8
        # errcheck16 = tf.abs(self.check_sel16 - self.g16_labels) #* mask16
        errcheck32 = tf.abs(self.check_sel32 - self.g32_labels)  # * mask32
        errcheck64 = tf.abs(self.check_sel64 - self.g64_labels)  # * mask64
        errcheck128 = tf.abs(self.check_sel128 - self.g_labels)  # * mask128

        self.weightedErrL1 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errL1, 1), 1))
        self.symErrL1 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(symL1(  # self.processor(self.G)
            tf.nn.avg_pool(self.G, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        ), 1), 1))
        self.weightedErrL2 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errL1_2, 1), 1))
        self.symErrL2 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(symL1(self.processor(self.G2)), 1), 1))
        self.weightedErrL3 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errL1_3, 1), 1))
        self.symErrL3 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(symL1(self.processor(self.G3, reuse=True)), 1), 1))

        cond_L12 = tf.abs(tf.image.resize_bilinear(self.G, [64, 64]) - tf.stop_gradient(self.G2))
        # self.condErrL12 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(cond_L12, 1), 1))
        # cond_L23 = tf.abs(tf.image.resize_bilinear(self.G2, [32,32]) - tf.stop_gradient(self.G3))
        # self.condErrL23 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(cond_L23, 1), 1))
        self.tv_loss = tf.reduce_mean(total_variation(self.G))
        # self.weightedErr_check8 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errcheck8, 1), 1))
        # self.weightedErr_check16 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errcheck16, 1), 1))
        # self.weightedErr_check32 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errcheck32, 1), 1))
        # self.weightedErr_check64 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errcheck64, 1), 1))
        # self.weightedErr_check128 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errcheck128, 1), 1))
        # mean = tf.reduce_mean(tf.reduce_mean(self.G, 1,keep_dims=True), 2, keep_dims=True)
        # self.stddev = tf.reduce_mean(tf.squared_difference(self.G, mean))

        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = params.L1_1_W * (self.weightedErrL1 + params.SYM_W * self.symErrL1) + params.L1_2_W * (
                self.weightedErrL2 + params.SYM_W * self.symErrL2) \
                      + params.L1_3_W * (self.weightedErrL3 + params.SYM_W * self.symErrL3)
        self.g_loss += params.BELTA_FEATURE * self.dv_loss + params.ALPHA_ADVER * self.g_loss_adver + params.IDEN_W * self.idenloss + self.tv_loss * params.TV_WEIGHT

        self.rot_loss = params.PART_W * (self.eyel_loss + self.eyer_loss + self.nose_loss + self.mouth_loss)
        # self.sel_loss = self.weightedErr_check32 + self.weightedErr_check64 + self.weightedErr_check128
        # # self.g_loss += self.sel_loss
        #
        # self.var_file = open('var_log.txt', mode='a')
        t_vars = [var for var in tf.trainable_variables() if 'FeatureExtractDeepFace' not in var.name \
                  and 'processor' not in var.name]

        def isTargetVar(name, tokens):
            for token in tokens:
                if token in name:
                    return True
            return False

        dec128toks = ['dec128', 'recon128', 'check_img128']
        self.d_vars = [var for var in t_vars if 'discriminatorLocal' in var.name]
        self.all_g_vars = [var for var in t_vars if 'discriminatorLocal' not in var.name]
        self.rot_vars = [var for var in t_vars if 'Rotator' in var.name]
        self.sel_vars = [var for var in t_vars if 'select' in var.name]
        self.dec_vars = [var for var in t_vars if 'decoder' in var.name and 'select' not in var.name]
        self.enc_vars = [var for var in t_vars if 'encoder' in var.name]
        self.pre_vars = [var for var in t_vars if 'FeaturePredict' in var.name]
        #
        self.se_vars = list(self.enc_vars);
        self.se_vars.extend(self.sel_vars)

        self.ed_vars = list(self.dec_vars);
        self.ed_vars.extend(self.enc_vars);
        self.ed_vars.extend(self.pre_vars);
        self.ed_vars.extend(self.rot_vars);
        self.ed_vars.extend(self.sel_vars)

        # self.rd_vars = list(self.dec_vars); self.rd_vars.extend([var for var in self.d_vars if isTargetVar(var.name, dec128toks)])

        # print("-----enc and dec ---->", map(lambda x:x.name, self.ed_vars), sep='\n', file=var_file)
        # print("-----enc and sel ---->", map(lambda x:x.name, self.se_vars), sep='\n',  file=var_file)
        # print("-----discrim ---->", map(lambda x:x.name, self.d_vars),sep='\n',  file=var_file)
        self.saver = tf.train.Saver(t_vars, max_to_keep=2)

    def get_data(self, params):
        """Train DCGAN"""
        # data = glob(os.path.join("./data", params.dataset, "*.jpg"))

        # np.random.shuffle(data)
        #params.sample_dir += '{:05d}'.format(random.randint(1, 100000))

        self.d_optim = tf.train.AdamOptimizer(params.learning_rate, beta1=params.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)

        self.g_dec_optim = tf.train.AdamOptimizer(params.learning_rate, beta1=params.beta1) \
            .minimize(self.g_loss, var_list=self.ed_vars)
        # g_enc_optim = tf.train.AdamOptimizer(params.learning_rate * 0.001, beta1=params.beta1) \
        #                  .minimize(self.g_loss, var_list=self.enc_vars)
        # s_optim = tf.train.AdamOptimizer(params.learning_rate, beta1=params.beta1) \
        #                   .minimize(self.sel_loss, var_list=self.se_vars)
        # g_sel_dec_optim = tf.train.RMSPropOptimizer(params.learning_rate) \
        #                 .minimize(self.g_loss + self.sel_loss + self.rot_loss, var_list=self.all_g_vars)
        self.rot_optim = tf.train.AdamOptimizer(params.learning_rate, beta1=params.beta1) \
            .minimize(self.rot_loss, var_list=self.rot_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.get_checkpoint:
            if self.load(params, params.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

    def train(self, params):
        counter = random.randint(1, 30)
        self.start_time = time.time()

        summary_writer = tf.summary.FileWriter(
            "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        loss_d = tf.Variable(0, dtype=tf.float32)
        loss_d_summ = tf.summary.scalar('D Loss', loss_d)
        loss_g = tf.Variable(0, dtype=tf.float32)
        loss_g_summ = tf.summary.scalar('G Loss', loss_g)

        sample_imagesT, filenamesT, sample_eyelT, sample_eyerT, sample_noseT, sample_mouthT, \
        sample_labelsT, sample_leyelT, sample_leyerT, sample_lnoseT, sample_lmouthT, sample_idenT = self.data.test_batch(
            params.batch_size * self.sample_run_num * params.RANK_MUL, Pose=params.RANGE)

        # append loss log to file
        print("start training!")
        for epoch in xrange(params.epoch):
            # data = glob(os.path.join("./data", params.dataset, "*.jpg"))
            batch_idxs = min(self.data.size, params.train_size)
            # print('data.size=', data.size, 'params.train_size=', params.train_size, 'batch_idxs=', batch_idxs)

            for idx in xrange(0, batch_idxs):
                # load data from MultiPIE
                batch_images_with_code, batch_labels, batch_masks, verify_images, verify_labels, \
                batch_pose, batch_iden, batch_landmarks, \
                batch_eyel, batch_eyer, batch_nose, batch_mouth, \
                batch_eyel_label, batch_eyer_label, batch_nose_label, batch_mouth_label \
                    = self.data.next_image_and_label_mask_batch(params.batch_size, imageRange=params.RANGE,
                                                                imageRangeLow=params.RANGE_LOW)
                # batch_images = batch_images_with_code[:,:,:,0:3] #discard codes
                if params.WITHOUT_CODEMAP:
                    batch_images_with_code = batch_images_with_code[..., 0:3]

                # needs self.G(needing images with code) and real images
                for _ in range(params.UPDATE_D):
                    # Update D network
                    _ = self.sess.run([self.d_optim, ],
                                      feed_dict={self.images_with_code: batch_images_with_code,
                                                 self.labels: batch_labels,
                                                 self.eyel: batch_eyel, self.eyer: batch_eyer,
                                                 self.nose: batch_nose, self.mouth: batch_mouth,
                                                 })
                for _ in range(params.UPDATE_G):
                    # Update G network
                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _ = self.sess.run([self.rot_optim, self.g_dec_optim, ],
                                      # _ = self.sess.run([g_sel_dec_optim],
                                      feed_dict={self.images_with_code: batch_images_with_code,
                                                 self.labels: batch_labels,
                                                 self.eyel: batch_eyel, self.eyer: batch_eyer,
                                                 self.nose: batch_nose, self.mouth: batch_mouth,
                                                 self.poselabels: batch_pose, self.idenlabels: batch_iden,
                                                 self.landmarklabels: batch_landmarks,
                                                 self.eyel_label: batch_eyel_label,
                                                 self.eyer_label: batch_eyer_label,
                                                 self.nose_label: batch_nose_label,
                                                 self.mouth_label: batch_mouth_label
                                                 })

                counter += 1
                print('.', end='')
                # sys.stdout.flush()
            if (epoch % params.save_epoch == 0):
                self.save(params, params.checkpoint_dir, counter)

            self.errD, self.err_total_G = self.evaluate(params, epoch, idx, batch_idxs, self.start_time, 'train',
                                                        batch_images_with_code, batch_eyel, batch_eyer, batch_nose,
                                                        batch_mouth,
                                                        batch_labels, batch_eyel_label, batch_eyer_label,
                                                        batch_nose_label,
                                                        batch_mouth_label, batch_iden);

            self.sess.run(loss_d.assign(self.errD))
            summary_writer.add_summary(self.sess.run(loss_d_summ), epoch)
            self.sess.run(loss_g.assign(self.err_total_G))
            summary_writer.add_summary(self.sess.run(loss_g_summ), epoch)

            print ("Epoch : ", epoch, " D loss : ", self.errD," G loss : ", self.err_total_G)
        summary_writer.flush()
        summary_writer.close()

    def test(self, params):
        self.sample_images_data, self.filenames, self.sample_eyel, self.sample_eyer, self.sample_nose, self.sample_mouth, \
        sample_labels, sample_leyel, sample_leyer, sample_lnose, sample_lmouth, sample_iden = self.data.test_batch(
            params.batch_size * self.sample_run_num)

        if params.WITHOUT_CODEMAP:
            self.sample_images_data = self.sample_images_data[..., 0:3]

        batchnum = 2
        savedtest = 0
        savedoutput = 0
        for i in range(batchnum):
            print('generating test result batch{}'.format(i))
            ind = (i * params.batch_size, (i + 1) * params.batch_size)
            # Save images
            self.samples = self.sess.run(
                self.sample_generator,
                feed_dict={self.sample_images: self.sample_images_data[ind[0]:ind[1], :, :, :],
                           self.eyel_sam: self.sample_eyel[ind[0]:ind[1], ...],
                           self.eyer_sam: self.sample_eyer[ind[0]:ind[1], ...],
                           self.nose_sam: self.sample_nose[ind[0]:ind[1], ...],
                           self.mouth_sam: self.sample_mouth[ind[0]:ind[1], ...]}
            )
        colorgt = self.sample_images_data[ind[0]:ind[1], :, :, 0:3]
        savedtest += save_images(colorgt,'./{}/'.format(params.sample_dir), isOutput=False,
                                 filelist=self.filenames[ind[0]:ind[1]])
        savedoutput += save_images(self.samples[5],'./{}/'.format(params.sample_dir), isOutput=True,
                                   filelist=self.filenames[ind[0]:ind[1]])
        print("[{} completed{} and saved {}.]".format(params.sample_dir, savedtest, savedoutput))

    def get_signiture(self):
        inputs_dict = {
            "sample_images": self.sample_images,
            "eyel_sam": self.eyel_sam,
            "eyer_sam": self.eyer_sam,
            "nose_sam": self.nose_sam,
            "mouth_sam": self.mouth_sam
        }

        outputs_dict = {
            "samples": self.sample_generator[0]
        }
        return inputs_dict, outputs_dict

    def processor(self, images, reuse=False):
        # accept 3 channel images, output orginal 3 channels and 3 x 4 gradient map-> 15 channels
        with tf.variable_scope("processor") as scope:
            if reuse:
                scope.reuse_variables()
            input_dim = images.get_shape()[-1]
            gradientKernel = gradientweight()
            output_dim = gradientKernel.shape[-1]
            #print("processor:", output_dim) #3
            k_hw = gradientKernel.shape[0]
            init = tf.constant_initializer(value=gradientKernel, dtype=tf.float32)
            w = tf.get_variable('w', [k_hw, k_hw, input_dim, output_dim],
                                initializer=init)
            conv = tf.nn.conv2d(images, w, strides=[1, 1, 1, 1], padding='SAME')
            # conv = conv * 2
            return concat([images, conv], 3)

    def FeaturePredict(self, featvec, reuse=False):
        with tf.variable_scope("FeaturePredict") as scope:
            if reuse:
                scope.reuse_variables()
            identitylogits = linear(Dropout(featvec, keep_prob=0.3, is_training=not self.testing), output_size=360,
                       scope='idenLinear',
                       bias_start=0.1, with_w=True)[0]
            return None, identitylogits, None

    def discriminatorLocal(self, params, images, reuse=False):
        with tf.variable_scope("discriminatorLocal") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(images, params.df_dim, name='d_h0_conv'))
            # 64
            h1 = lrelu(batch_norm(conv2d(h0, params.df_dim * 2, name='d_h1_conv'), name='d_bn1'))
            # 32
            h2 = lrelu(batch_norm(conv2d(h1, params.df_dim * 4, name='d_h2_conv'), name='d_bn2'))
            # 16
            h3 = lrelu(batch_norm(conv2d(h2, params.df_dim * 8, name='d_h3_conv'), name='d_bn3'))
            # #8x8
            h3r1 = resblock(h3, name="d_h3_conv_res1")
            h4 = lrelu(batch_norm(conv2d(h3r1, params.df_dim * 8, name='d_h4_conv'), name='d_bn4'))
            h4r1 = resblock(h4, name="d_h4_conv_res1")
            h5 = conv2d(h4r1, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='d_h5_conv')
            h6 = tf.reshape(h5, [params.batch_size, -1])
            # fusing 512 feature map to one layer prediction.
            return h6, h6  # tf.nn.sigmoid(h6), h6

    def decoder(self, params, feat128, feat64, feat32, feat16, feat8, featvec,
                g128_images_with_code, g64_images_with_code, g32_images_with_code,
                eyel, eyer, nose, mouth, c_eyel, c_eyer, c_nose, c_mouth, batch_size=10, name="decoder", reuse=False):
        sel_feat_capacity = params.gf_dim
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            initial_all = concat([featvec, self.z], 1)
            initial_8 = relu(tf.reshape(
                linear(initial_all, output_size=8 * 8 * params.gf_dim, scope='initial8', bias_start=0.1, with_w=True)[
                    0],
                [batch_size, 8, 8, params.gf_dim]))
            initial_32 = relu(
                deconv2d(initial_8, [batch_size, 32, 32, params.gf_dim // 2], d_h=4, d_w=4, name="initial32"))
            initial_64 = relu(deconv2d(initial_32, [batch_size, 64, 64, params.gf_dim // 4], name="initial64"))
            initial_128 = relu(deconv2d(initial_64, [batch_size, 128, 128, params.gf_dim // 8], name="initial128"))

            before_select8 = resblock(concat([initial_8, feat8], 3), k_h=2, k_w=2, name="select8_res_1")
            # selection T module
            reconstruct8 = resblock(resblock(before_select8, k_h=2, k_w=2, name="dec8_res1"), k_h=2, k_w=2,
                                    name="dec8_res2")

            # selection F module
            reconstruct16_deconv = relu(
                batch_norm(deconv2d(reconstruct8, [batch_size, 16, 16, params.gf_dim * 8], name="g_deconv16"),
                           name="g_bnd1"))
            before_select16 = resblock(feat16, name="select16_res_1")
            reconstruct16 = resblock(resblock(concat([reconstruct16_deconv, before_select16], 3), name="dec16_res1"),
                                     name="dec16_res2")

            reconstruct32_deconv = relu(
                batch_norm(deconv2d(reconstruct16, [batch_size, 32, 32, params.gf_dim * 4], name="g_deconv32"),
                           name="g_bnd2"))
            before_select32 = resblock(concat([feat32, g32_images_with_code, initial_32], 3), name="select32_res_1")
            reconstruct32 = resblock(resblock(concat([reconstruct32_deconv, before_select32], 3), name="dec32_res1"),
                                     name="dec32_res2")
            img32 = tf.nn.tanh(conv2d(reconstruct32, 3, d_h=1, d_w=1, name="check_img32"))

            reconstruct64_deconv = relu(
                batch_norm(deconv2d(reconstruct32, [batch_size, 64, 64, params.gf_dim * 2], name="g_deconv64"),
                           name="g_bnd3"))
            before_select64 = resblock(concat([feat64, g64_images_with_code, initial_64], 3), k_h=5, k_w=5,
                                       name="select64_res_1")
            reconstruct64 = resblock(resblock(concat([reconstruct64_deconv, before_select64,
                                                      tf.image.resize_bilinear(img32, [64, 64])], 3),
                                              name="dec64_res1"), name="dec64_res2")
            img64 = tf.nn.tanh(conv2d(reconstruct64, 3, d_h=1, d_w=1, name="check_img64"))

            reconstruct128_deconv = relu(
                batch_norm(deconv2d(reconstruct64, [batch_size, 128, 128, params.gf_dim], name="g_deconv128"),
                           name="g_bnd4"))
            before_select128 = resblock(concat([feat128, initial_128, g128_images_with_code], 3), k_h=7, k_w=7,
                                        name="select128_res_1")
            reconstruct128 = resblock(concat([reconstruct128_deconv, before_select128,
                                              self.partCombiner(eyel, eyer, nose, mouth),
                                              self.partCombiner(c_eyel, c_eyer, c_nose, c_mouth),
                                              tf.image.resize_bilinear(img64, [128, 128])], 3), k_h=5, k_w=5,
                                      name="dec128_res1")
            reconstruct128_1 = lrelu(
                batch_norm(conv2d(reconstruct128, params.gf_dim, k_h=5, k_w=5, d_h=1, d_w=1, name="recon128_conv"),
                           name="recon128_bnc"))
            reconstruct128_1_r = resblock(reconstruct128_1, name="dec128_res2")
            reconstruct128_2 = lrelu(
                batch_norm(conv2d(reconstruct128_1_r, params.gf_dim / 2, d_h=1, d_w=1, name="recon128_conv2"),
                           name="recon128_bnc2"))
            img128 = tf.nn.tanh(conv2d(reconstruct128_2, 3, d_h=1, d_w=1, name="check_img128"))

            return img128, img64, img32, img32, img32, img128, img64, img32

    def generator(self, images, batch_size, params, name="generator", reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            # imgs: input: IMAGE_SIZE x IMAGE_SIZE x params.CHANNEL
            # return labels: IMAGE_SIZE x IMAGE_SIZE x 3
            # U-Net structure, slightly different from the original on the location of relu/lrelu
            # 128x128
            c0 = lrelu(conv2d(images, params.gf_dim, k_h=7, k_w=7, d_h=1, d_w=1, name="g_conv0"))
            c0r = resblock(c0, k_h=7, k_w=7, name="g_conv0_res")
            c1 = lrelu(batch_norm(conv2d(c0r, params.gf_dim, k_h=5, k_w=5, name="g_conv1"), name="g_bnc1"))
            # 64x64
            c1r = resblock(c1, k_h=5, k_w=5, name="g_conv1_res")
            c2 = lrelu(batch_norm(conv2d(c1r, params.gf_dim * 2, name='g_conv2'), name="g_bnc2"))
            # 32x32
            c2r = resblock(c2, name="g_conv2_res")
            c3 = lrelu(batch_norm(conv2d(c2r, params.gf_dim * 4, name='g_conv3'), name="g_bnc3"))
            # 16x16
            c3r = resblock(c3, name="g_conv3_res")
            c4 = lrelu(batch_norm(conv2d(c3r, params.gf_dim * 8, name='g_conv4'), name="g_bnc4"))
            # 8x8
            c4r = resblock(c4, name="g_conv4_res")
            # c5 = lrelu(batch_norm(conv2d(c4r, params.gf_dim*8, name='g_conv5'),name="g_bnc5"))
            # #4x4
            # #2x2
            # c6r = resblock(c6,k_h=2, k_w=2, name="g_conv6_res")
            c4r2 = resblock(c4r, name="g_conv4_res2")
            c4r3 = resblock(c4r2, name="g_conv4_res3")
            c4r4 = resblock(c4r3, name="g_conv4_res4")
            c4r4_l = tf.reshape(c4r4, [batch_size, -1])
            c7_l = linear(c4r4_l, output_size=512, scope='feature', bias_start=0.1, with_w=True)[0]
            c7_l_m = tf.maximum(c7_l[:, 0:256], c7_l[:, 256:])

            return c0r, c1r, c2r, c3r, c4r4, c7_l_m

    def partRotator(self, images, name, params, batch_size=10, reuse=False):
        # HW 40x40, 32x40, 32x48
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            c0 = lrelu(conv2d(images, params.gf_dim, d_h=1, d_w=1, name="p_conv0"))
            c0r = resblock(c0, name="p_conv0_res")
            c1 = lrelu(batch_norm(conv2d(c0r, params.gf_dim * 2, name="p_conv1"), name="p_bnc1"))
            # down1
            c1r = resblock(c1, name="p_conv1_res")
            c2 = lrelu(batch_norm(conv2d(c1r, params.gf_dim * 4, name='p_conv2'), name="p_bnc2"))
            # down2
            c2r = resblock(c2, name="p_conv2_res")
            c3 = lrelu(batch_norm(conv2d(c2r, params.gf_dim * 8, name='p_conv3'), name="p_bnc3"))
            # down3 5x5, 4x5, 4x6
            c3r = resblock(c3, name="p_conv3_res")
            c3r2 = resblock(c3r, name="p_conv3_res2")

            shape = c3r2.get_shape().as_list()
            d1 = lrelu(
                batch_norm(deconv2d(c3r2, [shape[0], shape[1] * 2, shape[2] * 2, params.gf_dim * 4], name="p_deconv1"),
                           name="p_bnd1"))
            # up1
            after_select_d1 = lrelu(
                batch_norm(conv2d(concat([d1, c2r], axis=3), params.gf_dim * 4, d_h=1, d_w=1, name="p_deconv1_s"),
                           name="p_bnd1_s"))
            d1_r = resblock(after_select_d1, name="p_deconv1_res")
            d2 = lrelu(
                batch_norm(deconv2d(d1_r, [shape[0], shape[1] * 4, shape[2] * 4, params.gf_dim * 2], name="p_deconv2"),
                           name="p_bnd2"))
            # up2
            after_select_d2 = lrelu(
                batch_norm(conv2d(concat([d2, c1r], axis=3), params.gf_dim * 2, d_h=1, d_w=1, name="p_deconv2_s"),
                           name="p_bnd2_s"))
            d2_r = resblock(after_select_d2, name="p_deconv2_res")
            d3 = lrelu(
                batch_norm(deconv2d(d2_r, [shape[0], shape[1] * 8, shape[2] * 8, params.gf_dim], name="p_deconv3"),
                           name="p_bnd3"))
            # up3
            after_select_d3 = lrelu(
                batch_norm(conv2d(concat([d3, c0r], axis=3), params.gf_dim, d_h=1, d_w=1, name="p_deconv3_s"),
                           name="p_bnd3_s"))
            d3_r = resblock(after_select_d3, name="p_deconv3_res")

            check_part = tf.nn.tanh(conv2d(d3_r, 3, d_h=1, d_w=1, name="p_check"))

        return d3_r, check_part

    def partCombiner(self, eyel, eyer, nose, mouth):
        '''
        x         y
        43.5823   41.0000
        86.4177   41.0000
        64.1165   64.7510
        47.5863   88.8635
        82.5904   89.1124
        this is the mean locaiton of 5 landmarks
        '''
        eyel_p = tf.pad(eyel, [[0, 0], [int(41 - EYE_H / 2 - 1), int(IMAGE_SIZE - (41 + EYE_H / 2 - 1))],
                               [int(44 - EYE_W / 2 - 1), int(IMAGE_SIZE - (44 + EYE_W / 2 - 1))], [0, 0]])
        eyer_p = tf.pad(eyer, [[0, 0], [int(41 - EYE_H / 2 - 1), int(IMAGE_SIZE - (41 + EYE_H / 2 - 1))],
                               [int(86 - EYE_W / 2 - 1), int(IMAGE_SIZE - (86 + EYE_W / 2 - 1))], [0, 0]])
        nose_p = tf.pad(nose, [[0, 0], [int(65 - NOSE_H / 2 - 1), int(IMAGE_SIZE - (65 + NOSE_H / 2 - 1))],
                               [int(64 - NOSE_W / 2 - 1), int(IMAGE_SIZE - (64 + NOSE_W / 2 - 1))], [0, 0]])
        month_p = tf.pad(mouth, [[0, 0], [int(89 - MOUTH_H / 2 - 1), int(IMAGE_SIZE - (89 + MOUTH_H / 2 - 1))],
                                 [int(65 - MOUTH_W / 2 - 1), int(IMAGE_SIZE - (65 + MOUTH_W / 2 - 1))], [0, 0]])
        eyes = tf.maximum(eyel_p, eyer_p)
        eye_nose = tf.maximum(eyes, nose_p)
        return tf.maximum(eye_nose, month_p)

    def evaluate(self, params, epoch, idx, batch_idxs, start_time, mode,
                 batch_images_with_code, batch_eyel, batch_eyer, batch_nose, batch_mouth,
                 batch_labels, batch_eyel_label, batch_eyer_label, batch_nose_label, batch_mouth_label, batch_iden):
        errD = self.d_loss.eval({self.images_with_code: batch_images_with_code, self.labels: batch_labels,
                                 self.eyel: batch_eyel, self.eyer: batch_eyer, self.nose: batch_nose,
                                 self.mouth: batch_mouth, })

        errG_L = self.weightedErrL1.eval({self.images_with_code: batch_images_with_code, self.labels: batch_labels,
                                          self.eyel: batch_eyel, self.eyer: batch_eyer, self.nose: batch_nose,
                                          self.mouth: batch_mouth, })
        errG_L2 = self.weightedErrL2.eval({self.images_with_code: batch_images_with_code, self.labels: batch_labels,
                                           self.eyel: batch_eyel, self.eyer: batch_eyer, self.nose: batch_nose,
                                           self.mouth: batch_mouth, })
        errG_L3 = self.weightedErrL3.eval({self.images_with_code: batch_images_with_code, self.labels: batch_labels,
                                           self.eyel: batch_eyel, self.eyer: batch_eyer, self.nose: batch_nose,
                                           self.mouth: batch_mouth, })
        errG_adver = self.g_loss_adver.eval({self.images_with_code: batch_images_with_code,
                                             self.eyel: batch_eyel, self.eyer: batch_eyer, self.nose: batch_nose,
                                             self.mouth: batch_mouth, })
        errtv = self.tv_loss.eval({self.images_with_code: batch_images_with_code, self.labels: batch_labels,
                                   self.eyel: batch_eyel, self.eyer: batch_eyer, self.nose: batch_nose,
                                   self.mouth: batch_mouth, })

        errG_sym = self.symErrL1.eval({self.images_with_code: batch_images_with_code,
                                       self.eyel: batch_eyel, self.eyer: batch_eyer, self.nose: batch_nose,
                                       self.mouth: batch_mouth, })
        errG2_sym = self.symErrL2.eval({self.images_with_code: batch_images_with_code,
                                        self.eyel: batch_eyel, self.eyer: batch_eyer, self.nose: batch_nose,
                                        self.mouth: batch_mouth, })
        errG3_sym = self.symErrL3.eval({self.images_with_code: batch_images_with_code,
                                        self.eyel: batch_eyel, self.eyer: batch_eyer, self.nose: batch_nose,
                                        self.mouth: batch_mouth, })

        errcheck32 = 0  # self.weightedErr_check32.eval({self.images_with_code: batch_images_with_code, self.labels : batch_labels, })
        errcheck64 = 0  # self.weightedErr_check64.eval({self.images_with_code: batch_images_with_code, self.labels : batch_labels, })
        errcheck128 = 0  # self.weightedErr_check128.eval({self.images_with_code: batch_images_with_code, self.labels : batch_labels, })

        erreyel = self.eyel_loss.eval({self.eyel: batch_eyel, self.eyel_label: batch_eyel_label})
        erreyer = self.eyer_loss.eval({self.eyel: batch_eyel, self.eyer: batch_eyer, self.eyer_label: batch_eyer_label})
        errnose = self.nose_loss.eval({self.nose: batch_nose, self.nose_label: batch_nose_label})
        errmouth = self.mouth_loss.eval({self.mouth: batch_mouth, self.mouth_label: batch_mouth_label})
        erriden = self.idenloss.eval({self.images_with_code: batch_images_with_code, self.idenlabels: batch_iden, })

        if 'f' in params.MODE:
            errDv = self.dv_loss.eval({self.images_with_code: batch_images_with_code, self.labels: batch_labels,
                                       self.eyel: batch_eyel, self.eyer: batch_eyer, self.nose: batch_nose,
                                       self.mouth: batch_mouth, })
        else:
            errDv = 0
        err_total_G = params.L1_1_W * (errG_L + errG_sym * params.SYM_W) + params.L1_2_W * (
                errG_L2 + errG2_sym * params.SYM_W) + params.L1_3_W * (
                              errG_L3 + errG3_sym * params.SYM_W) \
                      + params.ALPHA_ADVER * errG_adver + errDv * params.BELTA_FEATURE + params.IDEN_W * erriden
        errfeat_total = errcheck32 + errcheck64 + errcheck128 + params.PART_W * (erreyel + erreyer + errnose + errmouth)

        return errD, err_total_G

    # DEEPFACE MODEL BEGINS---
    def loadDeepFace(self, DeepFacePath):
        if DeepFacePath is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "DeepFace.pickle")
            DeepFacePath = path
            logging.info("Load npy file from '%s'.", DeepFacePath)
        if not os.path.isfile(DeepFacePath):
            logging.error(("File '%s' not found. "), DeepFacePath)
            sys.exit(1)
        with open(DeepFacePath, 'rb') as file:
            if (sys.version_info.major == 2):
                self.data_dict = pickle.load(file)
            else:
                self.data_dict = pickle.load(file, encoding='iso-8859-1')
        print("Deep Face pickle data file loaded")

    def FeatureExtractDeepFace(self, images, name="FeatureExtractDeepFace", reuse=False):
        # Preprocessing: from color to gray(reduce_mean)
        print("Feature Extract Deep Face ")
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            conv1 = self._conv_layer(images, name='conv1')
            print(3, type(3))
            slice1_1, slice1_2 = tf.split(conv1, 2, 3)
            eltwise1 = tf.maximum(slice1_1, slice1_2)
            pool1 = tf.nn.max_pool(eltwise1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')
            conv2_1 = self._conv_layer(pool1, name='conv2_1')
            slice2_1_1, slice2_1_2 = tf.split(conv2_1, 2, 3)
            eltwise2_1 = tf.maximum(slice2_1_1, slice2_1_2)

            conv2_2 = self._conv_layer(eltwise2_1, name='conv2_2')
            slice2_2_1, slice2_2_2 = tf.split(conv2_2, 2, 3)
            eltwise2_2 = tf.maximum(slice2_2_1, slice2_2_2)

            res2_1 = pool1 + eltwise2_2

            conv2a = self._conv_layer(res2_1, name='conv2a')
            slice2a_1, slice2a_2 = tf.split(conv2a, 2, 3)
            eltwise2a = tf.maximum(slice2a_1, slice2a_2)

            conv2 = self._conv_layer(eltwise2a, name='conv2')
            slice2_1, slice2_2 = tf.split(conv2, 2, 3)
            eltwise2 = tf.maximum(slice2_1, slice2_2)

            pool2 = tf.nn.max_pool(eltwise2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')

            conv3_1 = self._conv_layer(pool2, name='conv3_1')
            slice3_1_1, slice3_1_2 = tf.split(conv3_1, 2, 3)
            eltwise3_1 = tf.maximum(slice3_1_1, slice3_1_2)

            conv3_2 = self._conv_layer(eltwise3_1, name='conv3_2')
            slice3_2_1, slice3_2_2 = tf.split(conv3_2, 2, 3)
            eltwise3_2 = tf.maximum(slice3_2_1, slice3_2_2)

            res3_1 = pool2 + eltwise3_2

            conv3_3 = self._conv_layer(res3_1, name='conv3_3')
            slice3_3_1, slice3_3_2 = tf.split(conv3_3, 2, 3)
            eltwise3_3 = tf.maximum(slice3_3_1, slice3_3_2)

            conv3_4 = self._conv_layer(eltwise3_3, name='conv3_4')
            slice3_4_1, slice3_4_2 = tf.split(conv3_4, 2, 3)
            eltwise3_4 = tf.maximum(slice3_4_1, slice3_4_2)

            res3_2 = res3_1 + eltwise3_4

            conv3a = self._conv_layer(res3_2, name='conv3a')
            slice3a_1, slice3a_2 = tf.split(conv3a, 2, 3)
            eltwise3a = tf.maximum(slice3a_1, slice3a_2)

            conv3 = self._conv_layer(eltwise3a, name='conv3')
            slice3_1, slice3_2 = tf.split(conv3, 2, 3)
            eltwise3 = tf.maximum(slice3_1, slice3_2)

            pool3 = tf.nn.max_pool(eltwise3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')

            conv4_1 = self._conv_layer(pool3, name='conv4_1')
            slice4_1_1, slice4_1_2 = tf.split(conv4_1, 2, 3)
            eltwise4_1 = tf.maximum(slice4_1_1, slice4_1_2)

            conv4_2 = self._conv_layer(eltwise4_1, name='conv4_2')
            slice4_2_1, slice4_2_2 = tf.split(conv4_2, 2, 3)
            eltwise4_2 = tf.maximum(slice4_2_1, slice4_2_2)

            res4_1 = pool3 + eltwise4_2

            conv4_3 = self._conv_layer(res4_1, name='conv4_3')
            slice4_3_1, slice4_3_2 = tf.split(conv4_3, 2, 3)
            eltwise4_3 = tf.maximum(slice4_3_1, slice4_3_2)

            conv4_4 = self._conv_layer(eltwise4_3, name='conv4_4')
            slice4_4_1, slice4_4_2 = tf.split(conv4_4, 2, 3)
            eltwise4_4 = tf.maximum(slice4_4_1, slice4_4_2)

            res4_2 = res4_1 + eltwise4_4

            conv4_5 = self._conv_layer(res4_2, name='conv4_5')
            slice4_5_1, slice4_5_2 = tf.split(conv4_5, 2, 3)
            eltwise4_5 = tf.maximum(slice4_5_1, slice4_5_2)

            conv4_6 = self._conv_layer(eltwise4_5, name='conv4_6')
            slice4_6_1, slice4_6_2 = tf.split(conv4_6, 2, 3)
            eltwise4_6 = tf.maximum(slice4_6_1, slice4_6_2)

            res4_3 = res4_2 + eltwise4_6

            conv4a = self._conv_layer(res4_3, name='conv4a')
            slice4a_1, slice4a_2 = tf.split(conv4a, 2, 3)
            eltwise4a = tf.maximum(slice4a_1, slice4a_2)

            conv4 = self._conv_layer(eltwise4a, name='conv4')
            slice4_1, slice4_2 = tf.split(conv4, 2, 3)
            eltwise4 = tf.maximum(slice4_1, slice4_2)

            conv5_1 = self._conv_layer(eltwise4, name='conv5_1')
            slice5_1_1, slice5_1_2 = tf.split(conv5_1, 2, 3)
            eltwise5_1 = tf.maximum(slice5_1_1, slice5_1_2)

            conv5_2 = self._conv_layer(eltwise5_1, name='conv5_2')
            slice5_2_1, slice5_2_2 = tf.split(conv5_2, 2, 3)
            eltwise5_2 = tf.maximum(slice5_2_1, slice5_2_2)

            res5_1 = eltwise4 + eltwise5_2

            conv5_3 = self._conv_layer(res5_1, name='conv5_3')
            slice5_3_1, slice5_3_2 = tf.split(conv5_3, 2, 3)
            eltwise5_3 = tf.maximum(slice5_3_1, slice5_3_2)

            conv5_4 = self._conv_layer(eltwise5_3, name='conv5_4')
            slice5_4_1, slice5_4_2 = tf.split(conv5_4, 2, 3)
            eltwise5_4 = tf.maximum(slice5_4_1, slice5_4_2)

            res5_2 = res5_1 + eltwise5_4

            conv5_5 = self._conv_layer(res5_2, name='conv5_5')
            slice5_5_1, slice5_5_2 = tf.split(conv5_5, 2, 3)
            eltwise5_5 = tf.maximum(slice5_5_1, slice5_5_2)

            conv5_6 = self._conv_layer(eltwise5_5, name='conv5_6')
            slice5_6_1, slice5_6_2 = tf.split(conv5_6, 2, 3)
            eltwise5_6 = tf.maximum(slice5_6_1, slice5_6_2)

            res5_3 = res5_2 + eltwise5_6

            conv5_7 = self._conv_layer(res5_3, name='conv5_7')
            slice5_7_1, slice5_7_2 = tf.split(conv5_7, 2, 3)
            eltwise5_7 = tf.maximum(slice5_7_1, slice5_7_2)

            conv5_8 = self._conv_layer(eltwise5_7, name='conv5_8')
            slice5_8_1, slice5_8_2 = tf.split(conv5_8, 2, 3)
            eltwise5_8 = tf.maximum(slice5_8_1, slice5_8_2)

            res5_4 = res5_3 + eltwise5_8

            conv5a = self._conv_layer(res5_4, name='conv5a')
            slice5a_1, slice5a_2 = tf.split(conv5a, 2, 3)
            eltwise5a = tf.maximum(slice5a_1, slice5a_2)

            conv5 = self._conv_layer(eltwise5a, name='conv5')
            slice5_1, slice5_2 = tf.split(conv5, 2, 3)
            eltwise5 = tf.maximum(slice5_1, slice5_2)
            pool4 = tf.nn.max_pool(eltwise5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')
            pool4_transposed = tf.transpose(pool4, perm=[0, 3, 1, 2])
            # pool4_reshaped = tf.reshape(pool4_transposed, shape=[tf.shape(pool4)[0],-1])
            fc1 = self._fc_layer(pool4_transposed, name="fc1")
            slice_fc1_1, slice_fc1_2 = tf.split(fc1, 2, 1)
            eltwise_fc1 = tf.maximum(slice_fc1_1, slice_fc1_2)

            return eltwise1, eltwise2, eltwise3, eltwise5, pool4, eltwise_fc1
        # DEEPFACE NET ENDS---

        # DEEPFACE OPS BEGINS---

    def _conv_layer(self, input_, output_dim=96,
                    k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
                    name="conv2d"):
        # Note: currently kernel size and input output channel num are decided by loaded filter weights.
        # only strides are decided by calling param.
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(input_, filt, strides=[1, d_h, d_w, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            return tf.nn.bias_add(conv, conv_biases)
            return conv

    def _fc_layer(self, bottom, name="fc1", num_classes=None):
        with tf.variable_scope(name) as scope:
            # shape = bottom.get_shape().as_list()
            if name == 'fc1':
                filt = self.get_fc_weight(name)
                bias = self.get_bias(name)
            reshaped_bottom = tf.reshape(bottom, [tf.shape(bottom)[0], -1])
            return tf.matmul(reshaped_bottom, filt) + bias

    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        # print('Layer name: %s' % name)
        # print('Layer shape: %s' % str(shape))

        var = tf.get_variable(name="filter", initializer=init, shape=shape)
        return var

    def get_bias(self, name, num_classes=None):
        bias_wights = self.data_dict[name][1]
        shape = self.data_dict[name][1].shape
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        return var

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        return var

    # DEEPFACE OPS ENDS---

    def save(self, params, checkpoint_dir, step):
        model_name = "DCGAN" + datetime.datetime.now().strftime("%Y%m%d") + ".model"
        model_dir = "%s_%s_%s_%s" % (params.dataset_name, params.batch_size, params.output_size, params.MODE)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print(" [*] Saving checkpoints...at step " + str(step))
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, params, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s_%s" % (params.dataset_name, params.batch_size, params.output_size, params.MODE)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False
