import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

#Chest X-Ray
def preprocess_chest_xray(img_path):
    img = keras_image.load_img(img_path, target_size=(150, 150), color_mode="grayscale")
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

#Brain Tumor
def preprocess_mri_brain_tumor(img_path):
    img = keras_image.load_img(img_path, target_size=(299, 299), color_mode="rgb")
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = xception_preprocess(img)
    return img

#Skin Cancer
def preprocess_skin_cancer(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224), color_mode="rgb")
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = mobilenet_preprocess(img)
    return img

#Bone Fracture
def preprocess_bone_fracture(img_path):
    img = keras_image.load_img(img_path, target_size=(180, 180), color_mode="rgb")
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

