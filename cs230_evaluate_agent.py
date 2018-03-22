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
                  
    
    def interactive_plot(self):
        def plot_sonar(sensor_inputs,ax):
            sensor_inputs *= 200 #outputs divide distance by 200
            ax.set_theta_zero_location('N')
            rlim = 30
            #Plot the sensor inputs
            sensor_inputs = np.minimum(sensor_inputs,rlim)
            ax.set_rlim(0,rlim)
            ax.set_yticks(range(0, rlim, 10))
            sensor_angles=np.linspace(-np.pi/2.0,np.pi/2.0,len(sensor_inputs))*-1
            bars = ax.bar(sensor_angles, sensor_inputs, width=5.0*np.pi/180.0, bottom=0.0)
            # Use custom colors and opacity
            for r, bar in zip(sensor_inputs, bars):
                bar.set_facecolor(plt.cm.jet_r(r/rlim/2))
                bar.set_alpha(0.9)
            
            #Plot the car
            theta = [np.deg2rad(i) for i in [-180,-150,-30,0,30,150,180]]
            r1 = abs(2.0/np.cos(theta))
            r2 = [0]*len(r1)
            ax.plot(theta, r1, color='k')
            ax.fill_between(theta, r1, r2, alpha=0.2,color='k')  
     
        print('INTERACTIVE PLOT START'.center(80,'='))
        while True:
            input_string = input('Type distance at which you would like to plot: ')
            if len(input_string) == 0:
                break
            tag = float(input_string)

            sorted_tags = sorted(range(len(self.obs)), key= lambda i: abs(self.roll_distance[i]-tag))
            closest_tag = sorted_tags[0]
            print('Plotting at distance %i'%(int(self.roll_distance[closest_tag])))
            img = None
            if self.config.vision:
                img = self.obs[closest_tag].img
            savefile = '../experiments/image_to_sonar/' + 'img2snr_snapshot_at_%i_m.png'%(int(self.roll_distance[closest_tag]))
            
            plt.figure(tag,figsize=(12,10))
            ax = plt.subplot(221)
            plt.imshow(np.reshape(img,[64,64,3]),origin='lower')
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_xticklines(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.get_yticklines(), visible=False)
            plt.xlabel('Camera') #1
            
            ax = plt.subplot(222, projection = 'polar')
            plot_sonar(np.reshape(self.obs[closest_tag].track,[19,]),ax)
            plt.xlabel('True sonar') #1
            
            
            ax = plt.subplot(223)
            plt.imshow(self.grayscales[closest_tag][:,:,:])
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_xticklines(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.get_yticklines(), visible=False)
            plt.xlabel('Stack of 3 observations (input)') #1
            
            ax = plt.subplot(224, projection = 'polar')
            plot_sonar(np.reshape(self.sonars[closest_tag],[19,]),ax)
            #Plot the steering direction
            steering_direction = self.actions[closest_tag]
            print('Sterring direction: ', steering_direction)
            theta= [0,np.deg2rad(steering_direction*90)]
            rlim = 25
            r  = [0,rlim*0.8]
            ax.plot(theta,r,c='k',lw='4')
            plt.xlabel('Predicted sonar with steering dir') #1
            
            plt.savefig(savefile,dpi=300, bbox_inches='tight')
            
            
            
            #plot(self.sonars[closest_tag],self.actions[closest_tag],img=img,savefile=savefile)
            #plot(self.obs[closest_tag].track,self.actions[closest_tag],img=img,savefile=savefile)
        return
    
    def sample_twenty(self):    
        roll_out_distances = []
        #Sample 10 episodes
        for i in range(20):
            self.sample_one()
            roll_out_distances.append(max(self.roll_distance))
        print(roll_out_distances)
        mean = np.mean(roll_out_distances)
        mini = np.min(roll_out_distances)
        maxi = np.max(roll_out_distances)
        std  = np.sqrt(np.var(roll_out_distances) / len(roll_out_distances))
        print('Min distance:'.ljust(15,'_')+"%i"%mini)
        print('Mean distance:'.ljust(15,'_')+"%i"%mean)
        print('Max distance:'.ljust(15,'_')+"%i"%maxi)
        print('Std distance:'.ljust(15,'_')+"%i"%std)
    
    
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
    

