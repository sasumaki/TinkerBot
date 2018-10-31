import numpy as np
import tensorflow as tf
import os
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time
import re
from urllib.parse import urlparse

class SuperbotEnv():
  

  def __init__(self, url, worker_id = -1, verbose=False, rewardCritic=None, mapping=False):
    capabilities = DesiredCapabilities().CHROME
    capabilities["pageLoadStrategy"] = "none"
    capabilities['loggingPrefs'] = { 'browser':'ALL' }
    self.mapping = mapping
    self.possible_actions = []
    if not mapping:
      f = open('hrefs.txt', 'r', encoding="utf-8")
      self.possible_actions = f.read().splitlines()
      f.close()
      self.observation_space = np.zeros((128, 128, 3), dtype=np.float32)
      self.action_space = np.zeros(len(self.possible_actions), dtype=np.float32)
      self.rewardCritic = rewardCritic

    self.url = url
    self.worker_idx = worker_id
    self.picname = 'pic' + str(worker_id) + '.png'
    self.verbose = verbose
    if worker_id > -1:
      self.driver = webdriver.Chrome(desired_capabilities=capabilities)


  def step(self, action):
    """
    click the defined "action" and take a save the following as new state.
    dynamic action space?
    """
    if self.mapping:
      elems = self.driver.find_elements_by_xpath("//a[@href]")
      for elem in elems:
        self.possible_actions.append(elem.get_attribute("href"))
    else:
      self.step_count += 1
      reward = 0
      
      selected_action = self.possible_actions[action]
      selector = 'a[href="/' + selected_action + '"]'
      response = None
      try:
        if (self.verbose == True):
          print('worker ', self.worker_idx, ' going to: ', selected_action)
        response = self.visit_or_click(click = selector)
      except Exception as e:
        lost =  not self.driver.current_url.startswith(self.url)
        if lost:
          return(self.state, reward, True, {})
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

      reward = self.rewardCritic.evaluate(self.state, response)
      if self.step_count > 50:
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

  def wait_for_stable_render(self):
    no_changes = 0
    changes = 0
    render_started_at = time.time()
    while time.time() <= (render_started_at + 3):
      before_size = len(self.driver.page_source)  #TODO: weird shit
      time.sleep(0.1)
      after_size = len(self.driver.page_source)

      if not before_size:
        print("before_size is nil, wat")
        continue

      if not after_size:
        print("after_size is nil, wat")
        continue      

      if before_size == after_size:
        no_changes = no_changes + 1
        if no_changes == 3:
          break
      else:
        changes = changes + 1
        if (self.verbose):
          print(f"..content changed {abs(after_size - before_size)} chars")
      
    
    render_stopped_at = time.time()
    was_unstable = (no_changes != 3)

    render_took = ((render_stopped_at - render_started_at) - (0.1 * no_changes))
    if render_took < 0 or changes == 0:
      render_took = 0.0 

    return (render_took, was_unstable)
  
  def visit_or_click(self, visit=None, click=None):
    if visit == None and click == None:
      raise "click or visit required"

    starting_url = self.driver.current_url
    visit_started_at = time.time()

    if visit:
      uri = visit
      url_without_protocol = visit.split("://")[-1]

      print(f"visiting: {uri}")
      #TODO: got https redirect?
      self.driver.get(uri)
    elif click:
      self.driver.execute_script('document.querySelectorAll(\'' + click + '\')[0].click()')
    else:
      raise "wtf"

    for i in range(3):  # This forces to block until first content has loaded :o
      if len(self.driver.page_source):
        break

    visit_stopped_at = time.time()
    visit_took = (visit_stopped_at - visit_started_at)
    render_took, was_unstable = self.wait_for_stable_render()
    total_took = (visit_took + render_took)
    stopped_url = self.driver.current_url
    if self.verbose:
      print(f"..completed as ",  "un" if was_unstable else "stable -", f" visit {visit_took:.2f}, render {render_took:.2f}")

    response = {
      "action": ('visit' if visit else 'click'),
      "transition": {
        "from_url": starting_url,
        "to_url": stopped_url
      },
      "duration": {
        "total": total_took,
        "visit": visit_took,
        "render": render_took,
        "unstable": was_unstable
      },
      "status": None,
      "requests": {}, #TODO: contains only failed ones
      "logs": {
        "severe": [],
        "warning": [],
        "debug": [],
        "info": []
      }
    }

    for log_entry in self.driver.get_log('browser'):
      if log_entry["level"] == "WARNING":
        response["logs"]["warning"] = log_entry.message
      if log_entry["level"] == "SEVERE":
        response["logs"]["severe"] = log_entry.message

        message_url, message = zip(log_entry.message.split(" - "))
        message_url_without_protocol = message_url.split("://")[-1]
        http_status = re.match(r"the server responded with a status of (\d+)/", message)
        print("http-status: ", http_status)
        # unless http_status
        #   puts "did not get http_status - log_entry: #{log_entry.inspect}"
        # end

        if visit:
          if message_url_without_protocol == url_without_protocol:
            response["status"] = http_status
          else:
            response["requests"][message_url] = {
              "status": http_status
            }

        elif click:
          #TODO: this is not verified, but should work with visit also (above)?
          if stopped_url == message_url:
            response["status"] = http_status
          else:
            response["requests"][message_url] = {
              "status": http_status
            }
        else:
          raise "wtf"

      if log_entry["level"] == "DEBUG":
        response["logs"]["debug"] = log_entry.message
      if log_entry["level"] == "INFO":
        response["logs"]["info"] = log_entry.message
      else:
        print(f"log_entry level unknown: {log_entry.level} -- {log_entry.inspect}")
        raise f"unknown log level: {log_entry.level}"
    if not response["status"]:
      response["status"] = 200
    return response

  