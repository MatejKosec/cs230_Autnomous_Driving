#Populate the replay buffer such that we can then generate the train, dev, test sets
#from it
import pickle
from skimage import color
from os import path
from matplotlib import pyplot as plt
from gym_torcs import TorcsEnv
from sample_agent import RandomAgent
import numpy as np
from replay_buffer import ReplayBuffer
from get_buffer import GetBuffer

#Get the replay buffer object as used in CS234 homeworks
def PopulateReplayBuffer(replay_size,episode_count=2,max_steps=15, history_length=4, replay_file='/data/replay_buffer.pkl',buffer=None):    
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
    agent = RandomAgent(1)  # steering only

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
                action = agent.act(ob, reward, done, vision)
                
                ob, reward, done, _ = env.step(action)
                #print(ob)
                print('Reward:', reward)
                total_reward += reward
                step += 1
                
                #plt.imshow(ob.img.reshape(64,64,3)[::-1,:,:])
                #ADD THE OBSERVATION TO THE BUFFER
                frame = ob.img.reshape(64,64,3)[::-1,:,:]
                frame = color.rgb2gray(frame)
                frame = frame.reshape((64,64,1))
                idx = buffer.store_frame(frame)
                buffer.store_effect(idx, action, reward, done)
                if done:
                    break
            #Summarise the session
            print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
            print("Total Step: " + str(step))
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
        buffer_train = PopulateReplayBuffer(50000,1000,200, 4, replay_file_train)
        buffer_dev   = PopulateReplayBuffer(3000,60,200, 4, replay_file_dev)
        buffer_test  = PopulateReplayBuffer(1000,60,200, 4, replay_file_test)
        
    elif a==1:
        train_buffer = GetBuffer(replay_file_train)
        buffer_train = PopulateReplayBuffer(50000,5,800, 4, replay_file_train,train_buffer)
        
    