import caffe

import numpy
import ctypes

def load(so):

    feature = ctypes.cdll.LoadLibrary(so)
    print feature







if __name__ == "__main__":
    path = "./feature_generator.so"
    load(path)
