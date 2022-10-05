from scipy.spatial import distance
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import base64
import os

load_dotenv()

model_url = os.getenv("FE_MODULE_URL")
IMAGE_SHAPE = (224, 224) 

layer = hub.KerasLayer(model_url, input_shape=IMAGE_SHAPE+(3,))
model = tf.keras.Sequential([layer])

def normalize_image(file_base64):
  file = Image.open(BytesIO(base64.b64decode(file_base64))).convert('L').resize(IMAGE_SHAPE)
  file = np.stack((file,)*3, axis=-1)
  file = np.array(file)/255.0
  return file

def extract(file):
  embedding = model.predict(file[np.newaxis, ...])
  efficient_net_np = np.array(embedding)
  flattended_feature = efficient_net_np.flatten()
  return flattended_feature
