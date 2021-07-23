"""Train the model"""
import tensorflow as tf

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

from models.models import DCGAN

from scripts.utils import Params
from scripts.create_dataset import MultiPIE

import argparse
import os
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
#tf.config.experimental.set_memory_growth(gpus[1],True)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=bool, default=True)
args = parser.parse_args()


def main(_):
    params = Params('config/params.json')

    if not os.path.exists(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    # if not os.path.exists(params.sample_dir):
    #     os.makedirs(params.sample_dir)

    with tf.Session() as sess:
        dcgan = DCGAN(sess, args.checkpoint, params, False)
        dcgan.get_data(params)
        dcgan.train(params)

    #tensorboard --logdir=fit/20210514-120150/ --host localhost --port 8088

if __name__ == '__main__':
    tf.app.run()
