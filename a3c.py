import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import threading
from SuperbotEnv import SuperbotEnv
import numpy as np
import multiprocessing
import argparse
from queue import Queue
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Run A3C algorithm on the game')

parser.add_argument('--lr', default=0.001,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=20, type=int,
                    help='How often to update the global model.')
parser.add_argument('--max-eps', default=1000, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99,
                    help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='/tmp/', type=str,
                    help='Directory in which you desire to save the model.')
args = parser.parse_args()

class ActorCriticModel(keras.Model):
  def __init__(self, state_size, action_size):
    super(ActorCriticModel, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.conv1 = layers.Conv2D(100, kernel_size=(32,32), input_shape=((1, 128, 128, 4)), strides=1, padding='valid', activation='relu')
    self.flatten = layers.Flatten()
    self.dense1 = layers.Dense(100, activation='relu')
    self.policy_logits = layers.Dense(action_size, activation="sigmoid")
    self.dense2 = layers.Dense(100, activation='relu')
    self.values = layers.Dense(1)

  def call(self, inputs):
    x = self.conv1(inputs)
    x1 = self.flatten(x)
    x2 = self.dense1(x1)
    logits = self.policy_logits(x2)
    v1 = self.dense2(inputs)
    values = self.values(v1)
    return logits, values

def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):

  if global_ep_reward == 0:
    global_ep_reward = episode_reward
  else:
    global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
  print(
      f"Episode: {episode} | "
      f"Moving Average Reward: {int(global_ep_reward)} | "
      f"Episode Reward: {int(episode_reward)} | "
      f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
      f"Steps: {num_steps} | "
      f"Worker: {worker_idx}"
  )
  result_queue.put(global_ep_reward)
  return global_ep_reward

class MasterAgent():
  def __init__(self):
    self.game_name = 'superbot'
    save_dir = args.save_dir
    self.save_dir = save_dir
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    env = SuperbotEnv("https://www.varusteleka.com")
    self.state_size = env.observation_space.shape
    self.action_size = env.action_space.shape[0]
    self.opt = tf.train.AdamOptimizer(args.lr, use_locking=True)
    print(self.state_size, self.action_size)

    self.global_model = ActorCriticModel(self.state_size, self.action_size)
    self.global_model(tf.convert_to_tensor(np.random.rand(1, self.state_size[0], self.state_size[1], self.state_size[2]), dtype=tf.float32))


  def train(self):

    res_queue = Queue()

    workers = [Worker(self.state_size,
                      self.action_size,
                      self.global_model,
                      self.opt, res_queue,
                      i, game_name=self.game_name,
                      save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())] #multiprocessing.cpu_count()

    for i, worker in enumerate(workers):
      print("Starting worker {}".format(i))
      worker.start()

    moving_average_rewards = [] 
    while True:
      reward = res_queue.get()
      if reward is not None:
        moving_average_rewards.append(reward)
      else:
        break
    [w.join() for w in workers]



class Memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []

  def store(self, state, action, reward):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)

  def clear(self):
    self.states = []
    self.actions = []
    self.rewards = []


class Worker(threading.Thread):
  global_episode = 0
  global_moving_average_reward = 0
  best_score = 0
  save_lock = threading.Lock()

  def __init__(self,
               state_size,
               action_size,
               global_model,
               opt,
               result_queue,
               idx,
               game_name='superbot',
               save_dir='/tmp'):
    super(Worker, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.result_queue = result_queue
    self.global_model = global_model
    self.opt = opt
    self.local_model = ActorCriticModel(self.state_size, self.action_size)
    self.worker_idx = idx
    self.game_name = game_name
    self.env = SuperbotEnv("https://www.varusteleka.com", idx)
    self.save_dir = save_dir
    self.ep_loss = 0.0

  def run(self):
    total_step = 1
    mem = Memory()
    while Worker.global_episode < args.max_eps:
      current_state = self.env.reset()
      mem.clear()
      ep_reward = 0.
      ep_steps = 0
      self.ep_loss = 0

      time_count = 0
      done = False
      while not done:
        logits, _ = self.local_model(
            tf.convert_to_tensor(current_state[None, :],
                                 dtype=tf.float32))
        probs = tf.nn.softmax(logits)
        action = np.random.choice(self.action_size, p=probs.numpy()[0])
        print('probs: ', probs[0], 'action: ', action)
        new_state, reward, done, _ = self.env.step(action)
        if done:
          reward = -1
        ep_reward += reward
        mem.store(current_state, action, reward)

        if time_count == args.update_freq or done:
          # Calculate gradient wrt to local model. We do so by tracking the
          # variables involved in computing the loss by using tf.GradientTape
          with tf.GradientTape() as tape:
            total_loss = self.compute_loss(done,
                                           new_state,
                                           mem,
                                           args.gamma)
          self.ep_loss += total_loss
          # Calculate local gradients
          grads = tape.gradient(total_loss, self.local_model.trainable_weights)
          # Push local gradients to global model
          self.opt.apply_gradients(zip(grads,
                                       self.global_model.trainable_weights))
          # Update local model with new weights
          self.local_model.set_weights(self.global_model.get_weights())

          mem.clear()
          time_count = 0

          if done:  # done and print information
            Worker.global_moving_average_reward = \
              record(Worker.global_episode, ep_reward, self.worker_idx,
                     Worker.global_moving_average_reward, self.result_queue,
                     self.ep_loss, ep_steps)
            # We must use a lock to save our model and to print to prevent data races.
            if ep_reward > Worker.best_score:
              with Worker.save_lock:
                print("Saving best model to {}, "
                      "episode score: {}".format(self.save_dir, ep_reward))
                self.global_model.save_weights(
                    os.path.join(self.save_dir,
                                 'model_{}.h5'.format(self.game_name))
                )
                Worker.best_score = ep_reward
            Worker.global_episode += 1
        ep_steps += 1

        time_count += 1
        current_state = new_state
        total_step += 1
    self.result_queue.put(None)

  def compute_loss(self,
                   done,
                   new_state,
                   memory,
                   gamma=0.99):
    if done:
      reward_sum = 0.  
    else:
      reward_sum = self.local_model(
          tf.convert_to_tensor(new_state[None, :],
                               dtype=tf.float32))[-1].numpy()[0]
    discounted_rewards = []
    for reward in memory.rewards[::-1]: 
      reward_sum = reward + gamma * reward_sum
      discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()

    logits, values = self.local_model(
        tf.convert_to_tensor(np.vstack(memory.states),
                             dtype=tf.float32))
    advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                            dtype=tf.float32) - values
    value_loss = advantage ** 2

    policy = tf.nn.softmax(logits)
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

    policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
                                                                 logits=logits)
    policy_loss *= tf.stop_gradient(advantage)
    policy_loss -= 0.01 * entropy
    total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
    return total_loss


if __name__ == '__main__':
  master = MasterAgent()
  master.train()

