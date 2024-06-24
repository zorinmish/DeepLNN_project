import PIL
import tensorflow as tf
from tensorflow import keras
import numpy as np
import image_modification
import cv2

def colorization(model, processed_image_path):
    """Apply colorization."""
    bw_image, lab = colorization_prepare_image(processed_image_path)
    L = colorization_resize_image(lab)
    ab = colorization_run_model(model, L)
    rgb_image = colorization_post_process_image(ab, bw_image, lab)
    cv2.imwrite(processed_image_path, rgb_image)

def colorization_prepare_image(image_path):
    # Load and prepare the image for the model.
    bw_image = cv2.imread(image_path)
    normalized = bw_image.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    return bw_image, lab

def colorization_resize_image(lab):
    # Resize the image and split it into L and A channels.
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    return L

def colorization_run_model(net, L):
    # Run the model on the input image.
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1,2,0))
    return ab

def colorization_post_process_image(ab, bw_image, lab):
    #Post process the image to produce the colorized image.
    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255.0 * colorized).astype("uint8")
    #rgb_image = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    return colorized