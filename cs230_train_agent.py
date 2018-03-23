from policy_gradient_agent import PG
from model_image_to_sonar import Model as ImageToSonar
from plot_sensors import plot
#Config of good version
import tensorflow as tf
from gym_torcs import TorcsEnv
from skimage import color
import numpy as np
import functools
import time
import random
import os 
from matplotlib import pyplot as plt

class config():
    #TORCS settings
    vision = True
    throttle = False
    env_name = 'TORCS'

    # output config
    restore_from_ckpt = True
    output_path  = "../experiments/policy_gradient_new_sensors_pretrained/"
    model_output = output_path + "model.weights/"
    best_model_output = output_path + "best_model.weights/"
    restore_model_path = "../experiments/policy_gradient_new_sensors/" + "best_model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path
    record_freq = 5
    summary_freq = 1

    # model and training config
    num_batches = 50 # number of batches trained on 
    batch_size = 200 # number of steps used to compute each policy update
    max_ep_len = 200 # maximum episode length
    learning_rate = 5e-3
    gamma         = 0.90
    # the discount factor
    use_baseline = True
    normalize_advantage=True
    # parameters for the policy and baseline models
    n_layers = 4
    layer_size = 18
    keep_prob = 0.9
    activation=tf.nn.leaky_relu

    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
    if max_ep_len < 0: max_ep_len = batch_size
    
class Config230(object):    
    batch_size = 64
    n_epochs = 2
    lr = 0.05
    n_test_samples = 10
    results_dir='../experiments/image_to_sonar_gradient_weighted/'

class Img2Snr(ImageToSonar):  
    def build(self, config):
        self.config = config
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_coupled_loss(self.pred)
        self.train_op = self.add_coupled_training_op(self.loss)
        self.saver = tf.train.Saver()
        
    def initialize(self,sess):
        print("Restoring the best model weights found on the dev set")
        self.saver.restore(sess, '../data/weights/predictor.weights')
        return
    
    def create_feed_dict(self, obs_batch,dsonar_batch=None):
        """Creates the feed_dict for one step of training. """
        feed_dict = {self.input_frame_placeholder: obs_batch #past
                     }
        if type(dsonar_batch) != type(None):
            weights = dsonar_batch
            feed_dict[self.weights_placeholder]= weights
        print('dSonar batch shape', dsonar_batch.shape)
        print('Obs batch shape', obs_batch.shape)
        return feed_dict
        
        
    def initialize_img_buffer(self):
        self.img_buffer = np.zeros([1,64,64,3], dtype=np.float32)
        self.step = 0
    def get_feed_dict(self,img):
        img = img.reshape(64,64,3)[::-1,:,:]
        img = color.rgb2gray(img)
        img = img.reshape((1,64,64,1))
        if self.step==1:
            self.img_buffer=np.tile(img,[1,1,1,3])
        else:
            self.img_buffer[0,:,:,2] = self.img_buffer[0,:,:,1]
            self.img_buffer[0,:,:,1] = self.img_buffer[0,:,:,0]
            self.img_buffer[0,:,:,0] = img[0,:,:,0]
        self.step +=1
        feed_dict = {self.input_frame_placeholder: self.img_buffer}
        return feed_dict

    def train_on_batch(self, sess, observations_batch, sonar_batch):
        """Perform one step of gradient descent on the provided batch of data. """
        n_minibatches = sonar_batch.shape[0] // self.config.batch_size + 1
        batches = list(range(n_minibatches))
        random.shuffle(batches)
        for i in batches:
            print('Processing batch %i of %i'%(i,n_minibatches))
            start =  self.config.batch_size*i
            end   =  min(self.config.batch_size*(i+1),sonar_batch.shape[0])
            indexes = list(range(start,end))
            random.shuffle(indexes)
            feed = self.create_feed_dict(observations_batch[indexes,:,:,:], sonar_batch[indexes,:])
            _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed)
            self.train_writer.add_summary(summary)
        return loss
        
    def predict(self,session,img):
        feed = self.get_feed_dict(img)
        predictions = session.run([self.pred], feed_dict=feed)[0]
        #print('Predictions: ',predictions)
        return predictions, np.reshape(self.img_buffer,[64,64,3])
    
    def add_coupled_loss(self, pred):
        weights = self.weights_placeholder #these are gradients from downstream
        coupled_loss = tf.reduce_mean(tf.reduce_sum(pred*weights,axis=1))
        tf.summary.scalar("coupled_loss", coupled_loss)
        self.summary_op = tf.summary.merge_all()
        return coupled_loss
    
    def add_coupled_training_op(self, coupled_loss):
        return tf.train.AdamOptimizer(self.config.lr).minimize(coupled_loss)
    
    def save(self, sess):
        if not os.path.exists('../data/weights_coupled/'):
            os.makedirs('../data/weights_coupled/')
        self.saver.save(sess, '../data/weights_coupled/predictor.weights')

class PGP(PG):
    def addSonar(self):
        print('='*80)
        print("ADDING the cs230 image to sonar model..."),
        start = time.time()
        self.sonar_config = Config230()
        self.sonar_model = Img2Snr(self.sonar_config)
        print('Model has {} parameters'.format(model.count_trainable_params()))
        #Setup variable initialization
        self.sonar_session = tf.Session()
        train_writer = tf.summary.FileWriter(os.path.join(self.sonar_config.results_dir, 'train_summaries'), self.sonar_session.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(self.sonar_config.results_dir, 'eval_summaries'), self.sonar_session.graph)
        self.sonar_model.add_writers(train_writer, eval_writer)
        self.sonar_model.initialize(self.sonar_session)
        self.sonar_model.initialize_img_buffer()
        print("took {:.2f} seconds\n".format(time.time() - start))
        print('='*80)
        
        
        
    def image_to_sonar(self, img):
        return self.sonar_model.predict(self.sonar_session,img)
        
    
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
                  ob = env.reset()
              else:
                  ob = env.reset(relaunch=True)
              sonar,grayscale = self.image_to_sonar(ob.img)
              sonar = np.reshape(sonar,[19])
              state = np.concatenate([sonar,np.array([ob.speedX,ob.speedY, ob.speedZ])],axis=0)
              obs, states, actions, rewards,sonars,grayscales,frames = [], [], [], [],[],[],[]
              
              img = ob.img.reshape(64,64,3)[::-1,:,:]
              img = color.rgb2gray(img)
              img = img.reshape((64,64,1))
              frame = np.tile(img,[1,1,3])
              
              episode_reward = 0
              
              for step in range(self.config.max_ep_len):
                states.append(state)
                obs.append(ob)
                sonars.append(sonar)
                grayscales.append(grayscale)
                frames.append(frame)
                #print('State', state)
                action = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder : np.reshape(states[-1],[1,self.observation_dim])})[0]
                ob, reward, done, info = env.step(action)
                #print('\n State track', state.track)   
                #print('\n State focus', state.focus)
                sonar,grayscale = self.image_to_sonar(ob.img)
                sonar = np.reshape(sonar,[19])
                state = np.concatenate([sonar,np.array([ob.speedX,ob.speedY, ob.speedZ])],axis=0)
                #Get the last 3 frames
                img = ob.img.reshape(64,64,3)[::-1,:,:]
                img = color.rgb2gray(img)
                img = img.reshape((64,64,1))
                frame[:,:,2] = frame[:,:,1]
                frame[:,:,1] = frame[:,:,0]
                frame[:,:,0] = img[:,:,0]
                
    
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
                #if (not num_episodes) and t == self.config.batch_size:
                #  break
          
              path = {"observation"    : np.array(states), 
                              "reward" : np.array(rewards), 
                              "action" : np.array(actions),
                              "frame"  : np.array(frames)}
              paths.append(path)
              episode += 1
              if num_episodes and episode >= num_episodes:
                break        
        finally:
            env.end()  # This is for shutting down TORCS
            print("Finished TORCS session".center(80,'='))
            #Plot some of the frames:
            self.grayscales = grayscales
            self.sonars=sonars
            self.obs = obs
            self.actions = actions
        return paths, episode_rewards, episode_roll_distances

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
          print()
          print('Running batch %i of %i'%(t,self.config.num_batches))
          print()
          # collect a minibatch of samples
          paths, total_rewards, rollout_distances = self.sample_path() 
          scores_eval = scores_eval + total_rewards
          observations = np.concatenate([path["observation"] for path in paths])
          actions = np.concatenate([path["action"] for path in paths])
          rewards = np.concatenate([path["reward"] for path in paths])
          frames = np.concatenate([path["frame"] for path in paths])
          # compute Q-val estimates (discounted future returns) for each time step
          returns = self.get_returns(paths)
          advantages = self.calculate_advantage(returns, observations)
          
          #Check if current model is best:
          if max(rollout_distances) > self.max_max_roll_distance:
              print('New best model found! Saving under: ', self.config.best_model_output)
              self.saver.save(self.sess, self.config.best_model_output)
              self.sonar_model.save(self.sonar_session)
              
          #########################################################################
          # Run training on both networks by passing gradients down to the image
          # to sonar network 
          #########################################################################
          # run training operations
          if self.config.use_baseline:
            self.update_baseline(returns, observations)
            [_,downstream_grads] = self.sess.run([self.train_op,self.inp_grad_loss], feed_dict={
                        self.observation_placeholder : observations, 
                        self.action_placeholder : actions, 
                        self.advantage_placeholder : advantages,
                        self.lr: self.learning_rate})
            #Concatenate sonar an dsonar 
            dsonar_batch = downstream_grads[0][:,:19]
            print('dsonar_batch: ', dsonar_batch)
            self.sonar_model.train_on_batch(self.sonar_session,frames, dsonar_batch)
      
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
        self.export_plot(scores_eval, "Score", config.env_name, self.config.plot_output)
    
if __name__ == '__main__':
    #Create the policy gradient actor
    driver = tf.Graph()
    print('Building the RL driver'.center(80,'='))
    try:
        with driver.as_default():
            model = PGP(config())
            model.initialize() #load the model from config
            print('Model has {} parameters'.format(model.count_trainable_params()))
        print 
        print('Building the sonar estimator'.center(80,'='))
        print
        sonar_graph = tf.Graph()
        with sonar_graph.as_default():
            model.addSonar();
        model.train()
        #Save the model
        model.save()
    except:
        raise

    finally:
        model.sonar_session.close()
    

