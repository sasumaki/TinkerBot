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
    self.default_reward = 0

  def evaluate(self, state, response):
    levels = response["transition"]["to_url"].split('/')
    route = 'pics/all/' + '/'.join(levels[:-1])
    if len(levels) > 1:
      pathlib.Path(route).mkdir(parents=True, exist_ok=True) 
    imroute = route + '/' +str(uuid4()) + ".png"
    imageio.imsave(imroute, np.squeeze(state))
    reward = response["duration"]["total"]
    
    return(reward)





