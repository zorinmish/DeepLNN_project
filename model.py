import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras

def colorization_load_model_and_points(prototxt_path, model_path, kernel_path):
    # Load the model and points from the specified paths.
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path)
    points = points.transpose().reshape(2, 313, 1, 1)
    return net, points

def colorization_set_netblobs(net, points):
    # Set blobs for the net layers.
    net.getLayer(net.getLayerId('class8_ab')).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]