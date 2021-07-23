# Utility functions: Params loading, logger tensorflow

import json
import logging
import os
import scipy.misc


class Params():
    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__

def save_images(images, image_path, suffix=None, isOutput=False, filelist = None):
    #if isOutput:
    #   images = mirrorLeftToFull(images)
    return imsave(images, image_path, suffix, isOutput, filelist)

def imsave(images, path, suffix=None, isOutput=False, filelist=None):
    num = images.shape[0]
    for i in range(num):
        if filelist is None:
            filename = path+str(i)
        else:
            filename = path+filelist[i][:-4] #discard .png
        if not isOutput:
            filename += '_test'
        if suffix is not None:
            filename += suffix
        filename += '.png'
        dirName = os.path.dirname(filename)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        if images.shape[-1] == 1:
            scipy.misc.imsave(filename,images[i,:,:,0])
        else:
            scipy.misc.imsave(filename,images[i,:,:,:])
    return num