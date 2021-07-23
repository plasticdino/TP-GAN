import argparse
import os
import json
import requests
from scripts.utils import Params
from scripts.create_dataset import MultiPIE
from scripts.utils import *
import numpy as np
import tensorflow as tf
import csv
from PIL import Image

'''
test from docker
'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=int, default=0)
parser.add_argument('--serving_url', type=str, default="")
args = parser.parse_args()

serving_endpoint = "http://localhost:8501/v1/models/tpgan:predict"

# sample_images_data = np.ones([4,128, 128, 3], dtype=np.float32)
# sample_eyel = np.ones([4, 40, 40, 3], dtype=np.float32)
# sample_eyer = np.ones([4, 40, 40, 3], dtype=np.float32)
# sample_nose = np.ones([4,32, 40, 3], dtype=np.float32)
# sample_mouth = np.ones([4, 32, 48, 3], dtype=np.float32)

def generate(sample_images_data, sample_eyel, sample_eyer, sample_nose, sample_mouth):
    inp = {
        "eyel_sam": sample_eyel.tolist(),
        "eyer_sam": sample_eyer.tolist(),
        "mouth_sam": sample_mouth.tolist(),
        "nose_sam": sample_nose.tolist(),
        "sample_images": sample_images_data.tolist()
    }
    x = {"signature_name": "serving_default", "inputs": np.array(inp).tolist()}
    headers = {"content-type": "application/json"}

    data = json.dumps(x)

    resp = requests.post(serving_endpoint, data=data, headers=headers)
    # print("r = ", resp.text)
    # print("resp = ", resp)
    if resp.status_code == 200:
        predictions = json.loads(resp.text)['outputs']
    else:
        predictions = None

    #print("prediction", np.array(predictions).shape)  # (4, 128, 128, 3)
    predictions = np.array(predictions)
    imsave(predictions, './samples/')

def main(params):
    data = MultiPIE(params, LOAD_60_LABEL=params.LOAD_60_LABEL,
                        RANDOM_VERIFY=params.RANDOM_VERIFY,
                        MIRROR_TO_ONE_SIDE=True,
                        testing=True)
    sample_images_data, _, sample_eyel, sample_eyer, sample_nose, sample_mouth, _, _, _, _, _, _ = data.test_batch()
    generate(sample_images_data, sample_eyel, sample_eyer, sample_nose, sample_mouth)

if __name__ == '__main__':

    params = Params('config/params.json')
    
    if not os.path.exists(params.sample_dir):
        os.makedirs(params.sample_dir)
        
    main(params)

