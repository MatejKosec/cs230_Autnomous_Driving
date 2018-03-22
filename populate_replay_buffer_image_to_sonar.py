#Populate the replay buffer such that we can then generate the train, dev, test sets
#from it
import pickle
from skimage import color
from os import path
from matplotlib import pyplot as plt
from gym_torcs import TorcsEnv
import numpy as np
from replay_buffer import ReplayBuffer
from get_buffer import GetBuffer
from policy_gradient_evaluate_agent import PGP, config

#Get the replay buffer object as used in CS234 homeworks
def PopulateReplayBuffer(replay_size,episode_count=2,max_steps=2000, history_length=3, replay_file='/data/replay_buffer.pkl',buffer=None):    
    if buffer==None:
        buffer = ReplayBuffer(replay_size, history_length)
        print('Created a new replay buffer')
    else:
        print('Reusing old buffer')
    
    #The settings for the TORCS environment
    vision = True #get visual inputs
    #initialize the RL vars
    reward = 0
    done = False
    step = 0
    
    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=False)
    print('Here',env.action_space)
    #Grab a randomly acting agent
    model = PGP(config())
    model.initialize()
    agent = model

    print("TORCS Experiment Start.")
    try:
        for i in range(episode_count):
            print("Episode : " + str(i))
            
            if np.mod(i, 3) == 0:
                # Sometimes you need to relaunch TORCS because of the memory leak error
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset()
            
            #Sample observations, rewards, and perform actions
            total_reward = 0.
            for j in range(max_steps):
                action,daction = agent.act(ob, reward, done, vision)
                
                ob, reward, done, _ = env.step(action)
                #print(ob)
                #print('Reward:', reward)
                total_reward += reward
                step += 1
                if step%100 == 0:
                    print('Step: ', step)
                #plt.imshow(ob.img.reshape(64,64,3)[::-1,:,:])
                #ADD THE OBSERVATION TO THE BUFFER
                frame = ob.img.reshape(64,64,3)[::-1,:,:]
                frame = color.rgb2gray(frame)
                frame = frame.reshape((64,64,1))
                sonar_dsonar = np.stack([ob.track,daction[0:19]])
                idx = buffer.store_frame(frame)
                buffer.store_effect(idx, sonar_dsonar, reward, done)
                if done or buffer.num_in_buffer >= max_steps:
                    break
            #Summar1ise the session
            print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
            print("Total Step: " + str(step))
            if buffer.num_in_buffer >= max_steps:
                break
            print("")
    finally:
        env.end()  # This is for shutting down TORCS
        print("Finished torcs session")
    with open(replay_file, 'wb') as w:
        #Dump the replay buffer to the file
        pickle.dump(buffer, w)
        
        
if __name__ == "__main__":
    replay_file_train = '../data/replay_buffer_train.pkl'
    replay_file_dev = '../data/replay_buffer_dev.pkl'
    replay_file_test = '../data/replay_buffer_test.pkl'
    a = 0
    if a==0:
        buffer_train = PopulateReplayBuffer(50000,40,100, 3, replay_file_train)
        #buffer_dev   = PopulateReplayBuffer(100,10,100, 3, replay_file_dev)
        #buffer_test  = PopulateReplayBuffer(100,10,100, 3, replay_file_test)
        
    elif a==1:
        train_buffer = GetBuffer(replay_file_train)
        buffer_train = PopulateReplayBuffer(20000,10,5000, 3, replay_file_train,train_buffer)
        
    