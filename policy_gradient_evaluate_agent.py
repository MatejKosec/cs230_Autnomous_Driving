from policy_gradient_agent import PG
from plot_sensors import plot
#Config of good version
import tensorflow as tf
from gym_torcs import TorcsEnv
import numpy as np
import functools
import time

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

class PGP(PG):
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
            state = np.concatenate([ob.track,np.array([ob.speedX,ob.speedY, ob.speedZ])],axis=0)
            obs, states, actions, rewards = [], [], [], []
            
            done = False #has the episode ended? 
            start_time = time.time()
            while not done and (time.time()-start_time<100):
                states.append(state)
                obs.append(ob)
                action = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder : np.reshape(states[-1],[1,self.observation_dim])})[0]
                ob, reward, done, info = env.step(action)
                state = np.concatenate([ob.track,np.array([ob.speedX,ob.speedY, ob.speedZ])],axis=0)
                
                actions.append(action)
                rewards.append(reward)
                roll_distance.append(env.distance_travelled)

        finally:
            env.end()  # This is for shutting down TORCS
            print("Finished TORCS session".center(80,'='))
            print('Final distance: ', roll_distance[-1],' [m]')
            print('END PLOTTING MODULE'.center(80,'='))
            #Plot some of the frames:
            self.obs = obs
            self.actions = actions
            self.roll_distance = roll_distance
            return
        
        
        
    
    def interactive_plot(self):
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
            savefile = self.config.output_path + 'snapshot_at_%i_m.png'%(int(self.roll_distance[closest_tag]))
            plot(self.obs[closest_tag].track,self.actions[closest_tag],img=img,savefile=savefile)
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
    model = PGP(config())
    model.initialize() #load the model from config
    model.sample_one()
    model.interactive_plot()
    #model.sample_twenty()

