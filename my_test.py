import tensorflow as tf

from models.models import DCGAN

from scripts.utils import Params
from scripts.create_dataset import MultiPIE

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=bool, default=True)
args = parser.parse_args()

'''
test directly from checkpoint
'''
def main(_):
    params = Params('config/params.json')

    if not os.path.exists(params.sample_dir):
        os.makedirs(params.sample_dir)

    with tf.Session() as sess:
        dcgan = DCGAN(sess, args.checkpoint, params, False)
        dcgan.get_data(params)
        dcgan.test(params)

#tensorboard --logdir=fit/20210526-003319/ --host localhost --port 8088

if __name__ == '__main__':
    tf.app.run()