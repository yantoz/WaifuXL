dependencies = ['torch', 'numpy', 'onnxruntime']

import os
import tempfile
import zipfile
import shutil

import numpy as np
import onnxruntime as ort

import torch.hub

import logging

#logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("waifuxl")

class _WaifuXL():

    def __init__(self, onnxFile):
        self.session = ort.InferenceSession(onnxFile)
        inputs = self.session.get_inputs()
        for i in range(len(inputs)):
            log.debug("Input[{}]: name={}, shape={}, type={}".format(i, inputs[i].name, inputs[i].shape, inputs[i].type))
        outputs = self.session.get_outputs()
        for i in range(len(outputs)):
            log.debug("Output[{}]: name={}, shape={}, type={}".format(i, outputs[i].name, outputs[i].shape, outputs[i].type))

    def __call__(self, image):
        if self.session is None:
            self.load()
        log.debug("Input: {}".format(image.shape))
        # add alpha channel
        output = np.insert(image, 3, 255, axis=2)
        # infer
        output = self.session.run(None, {'input': output})[0]
        # remove alpha channel
        output = output[:, :, 0:3]
        log.debug("Output: {}".format(output.shape))
        return output

def WaifuXL(progress=True):

    BASE_URL = "https://github.com/TheFutureGadgetsLab/WaifuXL/archive/refs/tags"
    VERSION = "1.5.0"

    TARGET = os.path.join(torch.hub.get_dir(), "WaifuXL", VERSION)
    FILENAME = "superRes.onnx"

    if not os.path.isfile(os.path.join(TARGET, FILENAME)):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'waifuxl.zip')
            url = "{}/v{}.zip".format(BASE_URL, VERSION)
            print("Downloading {} to {}".format(url, path)) 
            torch.hub.download_url_to_file(url, path, progress=progress)
            with zipfile.ZipFile(path, 'r') as zip:
                zip.extractall(tmp)
            os.makedirs(TARGET, exist_ok=True)
            shutil.copy(os.path.join(tmp, "WaifuXL-{}".format(VERSION), "public", "models", FILENAME), TARGET)
   
    return _WaifuXL(os.path.join(TARGET, FILENAME))
