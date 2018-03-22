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
from matplotlib import pyplot as plt

class config():
    #TORCS settings
    vision = True
    throttle = False
    env_name = 'TORCS'

    # output config
    restore_from_ckpt = True
    output_path  = "../experiments/policy_gradient_new_sensors/"
    model_output = output_path + "model.weights/"
    best_model_output = output_path + "best_model.weights/"
    restore_model_path = output_path + "best_model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path
    record_freq = 5
    summary_freq = 1

    # model and training config
    num_batches = 50 # number of batches trained on 
    batch_size = 800 # number of steps used to compute each policy update
    max_ep_len = 800 # maximum episode length
    learning_rate = 1e-2
    gamma         = 0.90
    # the discount factor
    use_baseline = True
    normalize_advantage=True
    # parameters for the policy and baseline models
    n_layers = 4
    layer_size = 18
    keep_prob = 0.9
    activation=tf.nn.relu

    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
    if max_ep_len < 0: max_ep_len = batch_size
    
class Config230(object):    
    batch_size = 128
    n_epochs = 2
    lr = 0.005
    n_test_samples = 10
    results_dir='../experiments/image_to_sonar_gradient_weighted/'

class Img2Snr(ImageToSonar):  
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
                     
        
    def predict(self,session,img):
        feed = self.get_feed_dict(img)
        predictions = session.run([self.pred], feed_dict=feed)[0]
        #print('Predictions: ',predictions)
        return predictions, np.reshape(self.img_buffer,[64,64,3])

class PGP(PG):
    def addSonar(self):
        print("Building the cs230 image to sonar model..."),
        start = time.time()
        self.sonar_model = Img2Snr(Config230())
        print('Model has {} parameters'.format(model.count_trainable_params()))
        #Setup variable initialization
        self.sonar_model.initialize_img_buffer()
        saver = tf.train.Saver()
        self.sonar_session = tf.Session()
        print("Restoring the best model weights found on the dev set")
        saver.restore(self.sonar_session, '../data/weights/predictor.weights')
        print("took {:.2f} seconds\n".format(time.time() - start))
        
        
        
    def image_to_sonar(self, img):
        return self.sonar_model.predict(self.sonar_session,img)
        
    def sample_one(self):
        """
        MODIFIED SAMPLING FOR TORCS!
        """
        print
        print('START PLOTTING MODULE'.center(80,'='))
        roll_distance = []
        print    
        print("TORCS Experiment Start".center(80,'='))
        env = TorcsEnv(vision=self.config.vision, throttle=self.config.throttle)
        try:
            ob = env.reset()
            sonar,grayscale = self.image_to_sonar(ob.img)
            sonar = np.reshape(sonar,[19])
            state = np.concatenate([sonar,np.array([ob.speedX,ob.speedY, ob.speedZ])],axis=0)
            obs, states, actions, rewards,sonars,grayscales = [], [], [], [],[],[]
            
            done = False #has the episode ended? 
            start_time = time.time()
            while not done and (time.time()-start_time<300):
                states.append(state)
                obs.append(ob)
                sonars.append(sonar)
                grayscales.append(grayscale)
                state = np.concatenate([sonar,np.array([ob.speedX,ob.speedY, ob.speedZ])],axis=0)
                
                
                action = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder : np.reshape(state,[1,self.observation_dim])})[0]
                ob, reward, done, info = env.step(action)
                sonar,grayscale = self.image_to_sonar(ob.img)
                sonar = np.reshape(sonar,[19])
                
                
                #print('Action: ', action)
                actions.append(action)
                rewards.append(reward)
                roll_distance.append(env.distance_travelled)
                #print('Roll distance: ', roll_distance)
        except: 
            raise

        finally:
            env.end()  # This is for shutting down TORCS
            print("Finished TORCS session".center(80,'='))
            print('Final distance: ', roll_distance[-1],' [m]')
            print('END PLOTTING MODULE'.center(80,'='))
            #Plot some of the frames:
            self.grayscales = grayscales
            self.sonars=sonars
            self.obs = obs
            self.actions = actions
            self.roll_distance = roll_distance
            return
                  
    
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
        model.sample_one()
        model.sample_twenty()
        model.interactive_plot()
    except:
        raise

    finally:
        model.sonar_session.close()
    

