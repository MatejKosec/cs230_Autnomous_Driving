# -*- coding: UTF-8 -*-
import os
import sys
import logging
import time
import numpy as np
import tensorflow as tf
import gym
import scipy.signal
import functools
import os
import time
import inspect
from utils.general import get_logger, Progbar, export_plot
from config import config
from gym_torcs import TorcsEnv
#import logz


def build_mlp(
          mlp_input, 
          output_size,
          scope, 
          n_layers=config.n_layers, 
          size=config.layer_size, 
          output_activation=None):
  '''
  Build a feed forward network (multi-layer-perceptron, or mlp) 
  with 'n_layers' hidden layers, each of size 'size' units.
  Use tf.nn.relu nonlinearity between layers. 
  Args:
          mlp_input: the input to the multi-layer perceptron
          output_size: the output layer size
          scope: the scope of the neural network
          n_layers: the number of layers of the network
          size: the size of each layer:
          output_activation: the activation of output layer
  Returns:
          The tensor output of the network
  
  '''
  
  with tf.variable_scope(scope):
      x  = tf.contrib.layers.fully_connected(inputs=mlp_input, num_outputs=size, activation_fn = tf.nn.leaky_relu)
      for i in range(1,n_layers):
          x = tf.contrib.layers.fully_connected(inputs=x, num_outputs=size, activation_fn = tf.nn.relu)
      y  = tf.contrib.layers.fully_connected(inputs=x, num_outputs=output_size, activation_fn = tf.nn.tanh)

  return y 


class PG(object):
  """
  Abstract Class for implementing a Policy Gradient Based Algorithm
  """
  def __init__(self, config, logger=None):
    """
    Initialize Policy Gradient Class
  
    Args:
            config: class with hyperparameters
            logger: logger instance from logging module

    """
    # directory for training outputs
    if not os.path.exists(config.output_path):
      os.makedirs(config.output_path)
            
    # store hyper-params
    self.config = config
    self.logger = logger
    if logger is None:
      self.logger = get_logger(config.log_path)
  
    # discrete action space or continuous action space
    self.discrete = False
    self.observation_dim = 19+3
    self.action_dim = 1 + int(self.config.throttle)
    self.learning_rate = self.config.learning_rate
  
    # build model
    self.build()
    
    #Enable saving the model
    self.saver = tf.train.Saver()
  
  
  def add_placeholders_op(self):
    """
    Adds placeholders to the graph
    Set up the observation, action, and advantage placeholder
    """
    
    print('Observation dim: ', self.observation_dim)
    print('Action dim: ', self.action_dim)
    self.observation_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,self.observation_dim])# TODO
    if self.discrete:
      self.action_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])
    else:
      self.action_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,self.action_dim])
      
    #Placeholder for learning rate
    self.lr = tf.placeholder(dtype=tf.float32, shape=[])
   
    # Define a placeholder for advantages
    self.advantage_placeholder = tf.placeholder(dtype=tf.float32, shape=[None])
    
  
  def build_policy_network_op(self, scope = "policy_network"):
    """
    Build the policy network, construct the tensorflow operation to sample 
    actions from the policy network outputs, and compute the log probabilities
    of the taken actions (for computing the loss later). These operations are 
    stored in self.sampled_action and self.logprob. Must handle both settings
    of self.discrete.

    """
    
    if self.discrete:
      #print(self.action_dim)
      action_logits = build_mlp(self.observation_placeholder,self.action_dim ,scope=scope) # TODO
      #print('logits:', action_logits)
      self.sampled_action = tf.squeeze(tf.multinomial(logits=action_logits,num_samples=1),axis=1)# TODO 
      self.logprob =-tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.action_placeholder, logits=action_logits) # TODO 
    else:
      action_means = build_mlp(self.observation_placeholder,self.action_dim,scope=scope)  # TODO 
      with tf.variable_scope(scope):
          log_std = tf.get_variable(name='log_std', dtype=tf.float32, shape=[self.action_dim]) #TODO
      self.sampled_action = action_means + tf.exp(log_std)*tf.random_normal(shape=[self.action_dim])   # TODO 
      distribution = tf.contrib.distributions.MultivariateNormalDiag(action_means,tf.exp(log_std))
      self.logprob = distribution.log_prob(self.action_placeholder)
      self.inp_grad_means = tf.gradients(action_means,[self.observation_placeholder])
            
  
  
  def add_loss_op(self):
    """
    Sets the loss of a batch, the loss is a scalar 
    Recall the update for REINFORCE with advantage:
    θ = θ + α ∇_θ log π_θ(s_t, a_t) A_t
    """

    #self.loss = -tf.reduce_sum(self.logprob*self.advantage_placeholder)# TODO
    self.loss = -tf.reduce_mean(self.logprob*self.advantage_placeholder)# TODO
  
    self.inp_grad_loss = tf.gradients(self.loss,[self.observation_placeholder])
    

  
  def add_optimizer_op(self):
    """
    Sets the optimizer using AdamOptimizer
    """
    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    
  
  def add_baseline_op(self, scope = "baseline"):
    """
    Build the baseline network within the scope

    In this function we will build the baseline network.
    Use build_mlp with the same parameters as the policy network to
    get the baseline estimate. You also have to setup a target
    placeholder and an update operation so the baseline can be trained.
    
    Args:
            scope: the scope of the baseline network
  
    """
    with tf.variable_scope(scope):
        self.baseline = tf.squeeze(build_mlp(self.observation_placeholder, 1, scope=scope),axis=1) # TODO
        #print 'Baseline:', self.baseline
        self.baseline_target_placeholder = tf.placeholder(dtype=tf.float32, shape=[None])# TODO
        baseline_loss = tf.losses.mean_squared_error(self.baseline_target_placeholder, self.baseline)
        self.update_baseline_op = tf.train.AdamOptimizer(self.lr).minimize(baseline_loss)
  
  def build(self):
    """
    Build model by adding all necessary variables

    """
  
    # add placeholders
    self.add_placeholders_op()
    # create policy net
    self.build_policy_network_op()
    # add square loss
    self.add_loss_op()
    # add optmizer for the main networks
    self.add_optimizer_op()
  
    if self.config.use_baseline:
      self.add_baseline_op()
  
  def initialize(self):
    """
    Assumes the graph has been constructed (have called self.build())
    Creates a tf Session and run initializer of variables

    """
    # create tf session
    self.sess = tf.Session()
    # tensorboard stuff
    self.add_summary()
    # initiliaze all variables
    init = tf.global_variables_initializer()
    if self.config.restore_from_ckpt:
        self.saver.restore(self.sess, self.config.restore_model_path)
        print("Model restored.")
    else:
        self.sess.run(init)
        print("Initialized a fresh session.")
  
  
  def add_summary(self):
    """
    Tensorboard stuff. 
    
    """
    # extra placeholders to log stuff from python
    self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
    self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
    self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")
    self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")
    
    
    self.avg_roll_distance_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_roll_distance")
    self.max_roll_distance_placeholder = tf.placeholder(tf.float32, shape=(), name="max_roll_distance")
    
  
    # extra summaries from python -> placeholders
    tf.summary.scalar("Avg_Reward", self.avg_reward_placeholder)
    tf.summary.histogram("Avg_Reward_Histogram", self.avg_reward_placeholder)
    tf.summary.scalar("Max_Reward", self.max_reward_placeholder)    
    tf.summary.scalar("Std_Reward", self.std_reward_placeholder)
    tf.summary.scalar("Eval_Reward", self.eval_reward_placeholder)
    tf.summary.scalar("Avg_roll_distance", self.avg_roll_distance_placeholder)
    tf.summary.scalar("Max_roll_distance", self.max_roll_distance_placeholder)
            
    # logging
    self.merged = tf.summary.merge_all()
    self.file_writer = tf.summary.FileWriter(self.config.output_path,self.sess.graph) 

  def init_averages(self):
    """
    Defines extra attributes for tensorboard.
    """
    self.avg_reward = 0.
    self.max_reward = 0.
    self.std_reward = 0.
    self.eval_reward = 0.
    self.max_roll_distance=0.
    self.max_max_roll_distance = 0. #keep track for saving model
    self.avg_roll_distance=0.
  

  def update_averages(self, rewards, scores_eval, rollout_distances):
    """
    Update the averages.
    Args:
            rewards: deque
            scores_eval: list
    """
    self.avg_reward = np.mean(rewards)
    self.max_reward = np.max(rewards)
    self.std_reward = np.sqrt(np.var(rewards) / len(rewards))
    self.max_max_roll_distance = max(self.max_roll_distance,np.max(rollout_distances))
    self.max_roll_distance = np.max(rollout_distances)
    self.avg_roll_distance = np.mean(rollout_distances)
    #Update the batch lengths as well
    self.config.batch_size = max(self.config.batch_size,min(int(self.max_roll_distance**2/80.0),2000))
    self.config.max_ep_len = self.config.batch_size
    
  
    if len(scores_eval) > 0:
      self.eval_reward = scores_eval[-1]
  
  
  def record_summary(self, t):
    """
    Add summary to tfboard
    """
  
    fd = {
      self.avg_reward_placeholder: self.avg_reward, 
      self.max_reward_placeholder: self.max_reward, 
      self.std_reward_placeholder: self.std_reward, 
      self.eval_reward_placeholder: self.eval_reward, 
      self.avg_roll_distance_placeholder: self.avg_roll_distance,
      self.max_roll_distance_placeholder : self.max_roll_distance
    }
    summary = self.sess.run(self.merged, feed_dict=fd)
    # tensorboard stuff
    self.file_writer.add_summary(summary, t)
  def act(self, ob, reward, done, vision):
      state = np.concatenate([ob.track,np.array([ob.speedX,ob.speedY, ob.speedZ])],axis=0)
      [action,daction]  = self.sess.run([self.sampled_action,self.inp_grad_means], feed_dict={self.observation_placeholder : np.reshape(state,[1,self.observation_dim])})
      action = action[0]
      daction = daction[0]
      return action, daction
  
  
  def sample_path(self, num_episodes = None):
    """
    MODIFIED FOR TORCS!
    Sample path for the environment.
  
    Args:
            num_episodes:   the number of episodes to be sampled 
              if none, sample one batch (size indicated by config file)
    Returns:
        paths: a list of paths. Each path in paths is a dictionary with
            path["observation"] a numpy array of ordered observations in the path
            path["actions"] a numpy array of the corresponding actions in the path
            path["reward"] a numpy array of the corresponding rewards in the path
        total_rewards: the sum of all rewards encountered during this "path"

    """
    episode = 0
    episode_rewards = []
    episode_roll_distances = []
    paths = []
    t = 0
    i = 0
    print    
    print("TORCS Experiment Start".center(80,'='))
    env = TorcsEnv(vision=self.config.vision, throttle=self.config.throttle)
    #print('Num episodes', num_episodes)
    print('Using a batch size of: ',self.config.batch_size)
    try:
        while (num_episodes or t < self.config.batch_size):
          i+=1
          print('t', t,'i',i)
          #Avoid a memory leak in TORCS by relaunching
          if np.mod(i,10)==0:
              state = env.reset()
          else:
              state = env.reset(relaunch=True)
          state = np.concatenate([state.track,np.array([state.speedX,state.speedY, state.speedZ])],axis=0)
          states, actions, rewards = [], [], []
          episode_reward = 0
          
          for step in range(self.config.max_ep_len):
            states.append(state)
            #print('State', state)
            action = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder : np.reshape(states[-1],[1,self.observation_dim])})[0]
            state, reward, done, info = env.step(action)
            #print('\n State track', state.track)   
            #print('\n State focus', state.focus)
            state = np.concatenate([state.track,np.array([state.speedX,state.speedY, state.speedZ])],axis=0)
            
            
    
            #print('State', state)
            #print('Reward', reward)
            #print('info', info)
            actions.append(action)
            rewards.append(reward)
            episode_reward += reward
            t += 1
            if (done or step == self.config.max_ep_len-1):
              episode_rewards.append(episode_reward)
              episode_roll_distances.append(env.distance_travelled)
              break
            if (not num_episodes) and t == self.config.batch_size:
              break
      
          path = {"observation"    : np.array(states), 
                          "reward" : np.array(rewards), 
                          "action" : np.array(actions)}
          paths.append(path)
          episode += 1
          if num_episodes and episode >= num_episodes:
            break        
    finally:
        env.end()  # This is for shutting down TORCS
        print("Finished TORCS session".center(80,'='))
    return paths, episode_rewards, episode_roll_distances
  
  
  def get_returns(self, paths):
    """
    Calculate the returns G_t for each timestep
  
    Args:
      paths: recorded sampled path.  See sample_path() for details.
  
    After acting in the environment, we record the observations, actions, and
    rewards. To get the advantages that we need for the policy update, we have
    to convert the rewards into returns, G_t, which are themselves an estimate
    of Q^π (s_t, a_t):
    
       G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T
    
    where T is the last timestep of the episode.
    """

    all_returns = []
    for path in paths:
      rewards = path["reward"]
      returns = [0]
      for r in reversed(rewards):
          returns.append(r + config.gamma*returns[-1])
      assert returns.pop(0) == 0
      returns = list(reversed(returns))
      #print('Returns:', returns)
      all_returns.append(returns)
    returns = np.concatenate(all_returns)
  
    return returns
  
  
  def calculate_advantage(self, returns, observations):
    """
    Calculate the advantage
    Args:
            returns: all discounted future returns for each step
            observations: observations
              Calculate the advantages, using baseline adjustment if necessary,
              and normalizing the advantages if necessary.
              If neither of these options are True, just return returns.

    """
    adv = returns

    if self.config.use_baseline:
        baseline_vals = self.sess.run(self.baseline, feed_dict={
                    self.observation_placeholder : observations})# TODO
        #print('Baseline vals:' ,baseline_vals.shape )
        #print('Advantage val:' ,adv.shape )
        adv = returns- baseline_vals
    if self.config.normalize_advantage:
        adv = (adv-np.mean(adv))/np.std(adv)
    
    return adv
  
  
  def update_baseline(self, returns, observations):
    """
    Update the baseline

    """

    self.sess.run(self.update_baseline_op, feed_dict={
                    self.observation_placeholder : observations, 
                    self.baseline_target_placeholder : returns,
                    self.lr : self.learning_rate})
  
  
  def train(self):
    """
    Performs training
    """
    
    last_eval = 0 
    last_record = 0
    scores_eval = []
    
    
    
    self.init_averages()
    scores_eval = [] # list of scores computed at iteration time
    
    # Update learning rate
    if self.max_roll_distance > 400.0:
        self.learning_rate = pow(self.learning_rate,0.9)
  
    for t in range(self.config.num_batches):
  
      # collect a minibatch of samples
      paths, total_rewards, rollout_distances = self.sample_path() 
      scores_eval = scores_eval + total_rewards
      observations = np.concatenate([path["observation"] for path in paths])
      actions = np.concatenate([path["action"] for path in paths])
      rewards = np.concatenate([path["reward"] for path in paths])
      # compute Q-val estimates (discounted future returns) for each time step
      returns = self.get_returns(paths)
      advantages = self.calculate_advantage(returns, observations)
      
      #Check if current model is best:
      if max(rollout_distances) > self.max_max_roll_distance:
          print('New best model found! Saving under: ', self.config.best_model_output)
          self.saver.save(self.sess, self.config.best_model_output)
          
      
      # run training operations
      if self.config.use_baseline:
        self.update_baseline(returns, observations)
      self.sess.run(self.train_op, feed_dict={
                    self.observation_placeholder : observations, 
                    self.action_placeholder : actions, 
                    self.advantage_placeholder : advantages,
                    self.lr: self.learning_rate})
  
      # tf stuff
      if (t % self.config.summary_freq == 0):
        self.update_averages(total_rewards, scores_eval, rollout_distances)
        self.record_summary(t)
      
      print("Learning rate:", self.learning_rate)
      # compute reward statistics for this batch and log
      avg_reward = np.mean(total_rewards)
      sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
      msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
      self.logger.info(msg)
      self.saver.save(self.sess, self.config.model_output)
      

      
    self.logger.info("- Training done.")
    export_plot(scores_eval, "Score", config.env_name, self.config.plot_output)


  def evaluate(self,num_episodes=1):
    """
    Evaluates the return for num_episodes episodes.
    Not used right now, all evaluation statistics are computed during training 
    episodes.
    """
    paths, rewards, total_rollout = self.sample_path(num_episodes)
    avg_reward = np.mean(rewards)
    sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
    msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
    self.logger.info(msg)
    return avg_reward
  def count_trainable_params(self):
      shapes = [functools.reduce(lambda x,y: x*y,variable.get_shape()) for variable in tf.trainable_variables()]
      return functools.reduce(lambda x,y: x+y, shapes)
     
  
  

  def run(self):
    """
    Apply procedures of training for a PG.
    """
    # initialize
    self.initialize()
    # train the model
    self.train()
  def save(self):
    """
    Saves session
    """
    if not os.path.exists(self.config.model_output):
        os.makedirs(self.config.model_output)

    self.saver.save(self.sess, self.config.model_output)
          
if __name__ == '__main__':
    #Create the policy gradient actor
    model = PG(config)
    #Train it
    model.run()
    #Evaluate it
    model.evaluate()
    #Save the model
    model.save()
