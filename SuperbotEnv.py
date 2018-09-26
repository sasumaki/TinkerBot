import numpy as np
import tensorflow as tf
import json
import requests
import os
from PIL import Image
from io import BytesIO
from selenium import webdriver
import re

class SuperbotEnv():

  def __init__(self, url, worker_id = -1):

    self.url = url
    self.worker_idx = worker_id
    self.picname = 'pic' + str(worker_id) + '.png'
    self.observation_space = np.zeros((128, 128, 4), dtype=np.float32)
    self.action_space = np.zeros(10, dtype=np.float32)


  def step(self, action):
    """
    click the defined "action" and take a save the following as new state.

    dynamic action space?
    """
    
    asd = self.possible_actions[action]
    selector = 'a[href="' + asd.split('.com')[1] + '"]'
    print('worker ', self.worker_idx, ' going to: ', asd)
    while True:
      try:
        self.driver.execute_script('document.querySelectorAll(\'' + selector + '\')[0].click()')
        break
      except:
        perkele = np.random.choice(self.possible_actions)
        selector = 'a[href="' + perkele.split('.com')[1] + '"]'
        self.driver.execute_script('document.querySelectorAll(\'' + selector + '\')[0].click()')
  
    self.possible_actions = []
    for element in self.driver.find_elements_by_css_selector("a"):
      href = element.get_attribute('href')
      if element.is_displayed() and re.match(r'^' + self.url + r'/', href) is not None:
        self.possible_actions.append(href)

    self.driver.save_screenshot(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pics', self.picname))
    self.state = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pics', self.picname)).resize((128,128), Image.ANTIALIAS)

    reward = 1
    done = False
    return(np.array(self.state), reward, done, {})

  def reset(self):
    self.driver = webdriver.Chrome()
    self.driver.get(self.url)
    self.possible_actions = []

    self.driver.save_screenshot(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pics', self.picname))
    self.state = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pics', self.picname)).resize((128,128), Image.ANTIALIAS)

    for element in self.driver.find_elements_by_css_selector("a"):
      href = element.get_attribute('href')
      if element.is_displayed() and re.match(r'^' + self.url + r'/', href) is not None:
        self.possible_actions.append(href)
    return(np.array(self.state))
