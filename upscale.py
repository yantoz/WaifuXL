import os
import sys
import torch
import numpy as np

from PIL import Image

class WaifuXL():
    def __init__(self):
        super().__init__()
        self.hub_repo = '.'
        self._model = None

    def load_model(self):
        return torch.hub.load(self.hub_repo, 'WaifuXL', source='local')

    def predict(self, img):
        h, w, d = img.shape
        assert d == 3, "Input image must be RGB"
        return self.model(img)

    @property
    def model(self):
        if self._model is None:
            self._model = self.load_model()
        return self._model

def upscaleFile(inFilename, outFilename):
    with open(inFilename, 'rb') as file:
        img = np.asarray(Image.open(file))
    model = WaifuXL()
    img = Image.fromarray(model.predict(img))
    img.save(outFilename)
    

if __name__ == '__main__':
    if len(sys.argv) < 3 or not os.path.isfile(sys.argv[1]):
        print("Usage: {} <infilename> <outfilename>".format(sys.argv[0]))
        sys.exit(-1)
    upscaleFile(sys.argv[1], sys.argv[2])
