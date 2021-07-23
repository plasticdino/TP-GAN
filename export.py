import tensorflow as tf

from models.models import DCGAN

from scripts.utils import Params
from scripts.create_dataset import MultiPIE

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=bool, default=True)

parser.add_argument('--out-path', type=str,
                        dest='out_path',
                        help='model output directory',
                        metavar='MODEL_OUT',
                        #required=True
                        default = 'export1'
                        ) #manually transfer to export/tpgan/{version}
args = parser.parse_args()


def main(_):
    params = Params('config/params.json')

    if not os.path.exists(params.sample_dir):
        os.makedirs(params.sample_dir)

    with tf.Session(graph=tf.Graph()) as sess:
        dcgan = DCGAN(sess, args.checkpoint, params, False)
        dcgan.get_data(params)
        print("Model restored.")
        inputs, outputs = dcgan.get_signiture()
        signature_def_map = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.saved_model.predict_signature_def(
                    inputs=inputs,
                    outputs=outputs
            )
        }

        builder = tf.saved_model.builder.SavedModelBuilder(args.out_path)
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.tag_constants.SERVING],
                                             strip_default_attrs=True,
                                             signature_def_map=signature_def_map
                                             )
        builder.save()

if __name__ == '__main__':
    tf.app.run()
    #saved_model_cli show --dir export/tpgan/1 --all
    #docker run -p 8501:8501 -v /d/I3/internship/export/tpgan:/models/tpgan -e MODEL_NAME=tpgan tensorflow/serving