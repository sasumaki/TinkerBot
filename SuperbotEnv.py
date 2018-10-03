import numpy as np
import tensorflow as tf
import os
from PIL import Image
from io import BytesIO
from selenium import webdriver
import re

class SuperbotEnv():

  def __init__(self, url, worker_id = -1, verbose=False, rewardCritic=None):
    f = open('hrefs.txt', 'r', encoding="utf-8")
    self.possible_actions = f.read().splitlines()
    f.close()
    self.url = url
    self.worker_idx = worker_id
    self.picname = 'pic' + str(worker_id) + '.png'
    self.observation_space = np.zeros((128, 128, 3), dtype=np.float32)
    self.action_space = np.zeros(len(self.possible_actions), dtype=np.float32)
    self.verbose = verbose
    if worker_id > -1:
      self.driver = webdriver.Chrome()
    self.rewardCritic = rewardCritic


  def step(self, action):
    """
    click the defined "action" and take a save the following as new state.
    dynamic action space?
    """
    self.step_count += 1
    reward = 1

    selected_action = self.possible_actions[action]
    selector = 'a[href="/' + selected_action + '"]'
    try:
      if (self.verbose == True):
        print('worker ', self.worker_idx, ' going to: ', selected_action)
      self.driver.execute_script('document.querySelectorAll(\'' + selector + '\')[0].click()')
    except Exception as e:
      if (self.verbose == True):
        print(e)
      return(self.state, reward, False, {})
        

    # self.state_possible_actions = []
    # for element in self.driver.find_elements_by_css_selector("a"):
    #   href = element.get_attribute('href')
    #   if element.is_displayed() and re.match(r'^' + self.url + r'/', href) is not None:
    #     self.state_possible_actions.append(href)

    self.driver.save_screenshot(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pics', self.picname))
    self.state = np.array(Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pics', self.picname)).resize((128,128), Image.ANTIALIAS).convert('RGB')).reshape(1,128,128,3)
    lost = "https://www.varusteleka.com" not in self.driver.current_url

    reward = self.rewardCritic.evaluate(self.state, selected_action)
    if self.step_count > 50 or lost == True:
      if lost == True:
        reward = 0
      done = True
    else:
      done = False
    return(self.state, reward, done, {})

  def reset(self):
    self.step_count = 0
    self.driver.get(self.url)
    self.state_possible_actions = []
    self.driver.save_screenshot(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pics', self.picname))
    self.state = np.array(Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pics', self.picname)).resize((128,128), Image.ANTIALIAS).convert('RGB')).reshape(1,128,128,3)


    return(self.state)
