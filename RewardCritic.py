import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import imageio
import pathlib
from uuid import uuid4
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class RewardCritic(keras.Model):
  """
  supposed to check if given state is broken or revisited and give reward depending on that
  """
  def __init__(self, default_reward=0):
    super(RewardCritic, self).__init__()
    self.conv1 = layers.Conv2D(3, kernel_size=(3,3), input_shape=((1, 128, 128, 3)), activation='relu', data_format="channels_last")
    self.maxpool = layers.MaxPool2D()
    self.conv2 = layers.Conv2D(3, kernel_size=(3,3), input_shape=((1, 128, 128, 3)), activation='relu', data_format="channels_last")
    self.flatten = layers.Flatten()
    self.dense1 = layers.Dense(256, activation='relu')
    self.dense2 = layers.Dense(32, activation='relu')
    self.default_reward = default_reward
    self.error_images = []
    self.error_images.append((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pics', '404.png')))
    self.error_encoded = self.encode(np.array(Image.open(self.error_images[0]).resize((128,128), Image.ANTIALIAS).convert('RGB')).reshape(1,128,128,3))

  def evaluate(self, state, url):
    levels = url.split('/')
    route = 'pics/all/' + '/'.join(levels[:-1])
    print(route)
    if len(levels) > 1:
      pathlib.Path(route).mkdir(parents=True, exist_ok=True) 
    imroute = route + '/' +str(uuid4()) + ".png"
    imageio.imsave(imroute, np.squeeze(state))
    similarity = cosine_similarity(state.reshape(1,-1), np.array(Image.open(self.error_images[0]).resize((128,128), Image.ANTIALIAS).convert('RGB')).reshape(1,-1))
    reward = self.default_reward
    if similarity > 0.999:
      reward = 1
    return(reward)
  
  def encode(self, image):
    r1 = self.conv1(tf.convert_to_tensor(image, dtype=tf.float32))
    r2 = self.maxpool(r1)
    r3 = self.conv2(r2)
    r4 = self.flatten(r3)
    r5 = self.dense1(r4)
    r6 = self.dense2(r5)
    return(r6)




