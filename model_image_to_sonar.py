import tensorflow as tf
import numpy as np
import functools
import os

#Model skeleton taken from CS224N    
#This moodel takes as an input a certain number of past states and 
#attempts to predict what the current sonar is measuring
class Model(object):
    def __init__(self,config):
        self.build(config)
        
    def add_writers(self, train_writer, eval_writer):
        self.train_writer = train_writer
        self.eval_writer = eval_writer
        print('Summary writers add to the graph')
    
    def add_placeholders(self):
        
        self.sonar_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,19]) #value from 0 to 1
        self.weights_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,19]) #value from 0 to 1
        self.input_frame_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,64,64,3]) #grayscale frames
        

    def create_feed_dict(self, obs_batch,sonar_batch=None):
        """Creates the feed_dict for one step of training. """
        feed_dict = {self.input_frame_placeholder: obs_batch #past
                     }
        if type(sonar_batch) != type(None):
            sonar = sonar_batch[:,:19]
            weights = sonar_batch[:,19:]
            feed_dict[self.sonar_placeholder]= sonar
            feed_dict[self.weights_placeholder]= weights
            #print('Sonar batch shape', sonar_batch.shape)
            #print('Obs batch shape', obs_batch.shape)
        return feed_dict

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions. """
        #BASED ON THE SEGNET ARCHITECTURE http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7803544
        
        #Convolution, convolution, pool
        x1 = tf.contrib.layers.conv2d(self.input_frame_placeholder, 32, \
                                          kernel_size=[4,4], stride=(1,1), padding='SAME', normalizer_fn=tf.contrib.layers.batch_norm)
        print('x1 shape', x1.shape)
        x1 = tf.contrib.layers.conv2d(x1, 24, \
                                          kernel_size=[4,4], stride=(1,1), padding='SAME', normalizer_fn=tf.contrib.layers.batch_norm)
        print('x1 shape', x1.shape)
        x2 = tf.contrib.layers.max_pool2d(x1, \
                                          kernel_size=[2,2], stride=(2,2), padding='SAME')
        print('x2  shape', x2.shape)
        x4=x2
                
        #Deconvolve, deconvolve
        x5 = x4
        x6= tf.contrib.layers.conv2d_transpose(x5, 12 , [4,4], stride=1, padding='SAME') #deconvolve
        print('x6 shape', x6.shape)
        x6= tf.contrib.layers.conv2d_transpose(x6, 12 , [4,4], stride=1, padding='SAME') #deconvolve
        print('x6 shape', x6.shape)
        
        #Upscale, deconvolve, deconvolve
        x7 = tf.contrib.layers.max_pool2d(x6, \
                                          kernel_size=[2,2], stride=(2,2), padding='SAME')
        print('x7 shape', x7.shape)
        x8= tf.contrib.layers.conv2d_transpose(x7, 12 ,  [4,4], stride=(1,1), padding='SAME',activation_fn=tf.nn.relu)
        print('x8 shape', x8.shape)
        x9 = tf.contrib.layers.conv2d(x8, 19, [16,16], stride=(1), padding='valid',activation_fn=tf.nn.relu)
        pred = tf.contrib.layers.conv2d(x9, 19, [1,1], stride=(1), padding='valid',activation_fn=tf.nn.sigmoid)
        print('pred shape', pred.shape)
        pred = tf.reshape(pred,[-1,19])
        print('pred shape', pred.shape)
        #Predictions should be between 0 and 1
        #raise(ValueError)
        return pred
        

    def add_loss_op(self, pred):
        weights = tf.abs(self.weights_placeholder)
        weights = weights/tf.reshape(tf.reduce_sum(weights, axis=1),[-1,1])*39.0/19.0
        #weights=(2-tf.cos(tf.linspace(-np.pi,np.pi,19)))
        loss = tf.reduce_mean(weights*(pred-self.sonar_placeholder)**2)
        
        tf.summary.scalar("loss", loss)
        self.summary_op = tf.summary.merge_all()
        return loss

    def add_training_op(self, loss):
        self.global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
        return tf.train.AdamOptimizer(self.config.lr).minimize(loss,global_step=self.global_step)        

    def train_on_batch(self, sess, observations_batch, sonar_batch):
        """Perform one step of gradient descent on the provided batch of data. """
        feed = self.create_feed_dict(observations_batch, sonar_batch)
        _, loss, summary, global_step = sess.run([self.train_op, self.loss, self.summary_op,self.global_step], feed_dict=feed)
        self.train_writer.add_summary(summary, global_step=global_step)
        return loss

    def loss_on_batch(self, sess, observations_batch, sonar_batch):
        """Make predictions for the provided batch of data """
        feed = self.create_feed_dict(observations_batch, sonar_batch)
        loss, summary, global_step = sess.run([self.loss,self.summary_op,self.global_step], feed_dict=feed)
        self.eval_writer.add_summary(summary, global_step=global_step)
        return loss

    def build(self, config):
        self.config = config
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        
    def compare_outputs(self, test_buffer, sess):
        
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
                
                
        samples = test_buffer.sample(self.config.n_test_samples)
        obs_batch, sonar_batch, rew_batch, next_obs_batch, done_mask = samples
        feed = self.create_feed_dict(obs_batch)
        predictions = sess.run([self.pred], feed_dict=feed)[0]
        print('Predictions shape:', predictions.shape)
        from matplotlib import pyplot as plt
        for i in range(self.config.n_test_samples):
            truth = sonar_batch[i,:]
            img   = obs_batch[i,:,:,-1]
            print('Truth shape:', truth.shape)
            print('Truth:', truth)
            prediction = predictions[i,:]
            print('Prediction shape:', prediction.shape)
            print('Prediction:', prediction)
            
            plt.figure(i,figsize=(14,5))
            #plt.title('Comparison between prediction and truth (test set)')
            #plot the current observation
            ax = plt.subplot(131)
            plt.imshow(img,cmap="Greys")
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_xticklines(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.get_yticklines(), visible=False)
            plt.xlabel('Greyscale observation') #1
            

            ax = plt.subplot(132, projection = 'polar')
            plot_sonar(np.reshape(prediction,[19,]),ax)
            plt.xlabel('Predicted sonar') #1
            
            ax = plt.subplot(133, projection = 'polar')
            plot_sonar(np.reshape(truth,[19,]),ax)
            plt.xlabel('True sonar') #1
            plt.savefig('../data/sonar_predicted_vs_true%i.png'%i,dpi=300, bbox_inches='tight')

                

                
        
    def run_epoch(self, sess, train_buffer, dev_buffer):
        n_minibatches = train_buffer.num_in_buffer // self.config.batch_size
        prog = tf.keras.utils.Progbar(target=n_minibatches)
        for i in range(n_minibatches):
            obs_batch, sonar_batch, rew_batch, next_obs_batch, done_mask = train_buffer.sample(self.config.batch_size)
            loss = self.train_on_batch(sess, obs_batch, sonar_batch)
            print('i',i)
            prog.update(i + 1, [("train loss", loss)], force=i + 1 == n_minibatches)
            if i%100 == 0 or i== n_minibatches -1:
                if i== n_minibatches -1: print("Evaluating on dev set",) 
                obs_batch, sonar_batch, rew_batch, next_obs_batch, done_mask = dev_buffer.sample(dev_buffer.num_in_buffer-1)
                dev_loss = self.loss_on_batch(sess, obs_batch,sonar_batch)
                if i==n_minibatches -1: print("Dev loss: {:.7f}".format(dev_loss))
        return dev_loss

    def fit(self, sess, saver,train_buffer, dev_buffer):
        best_dev_loss = 100
        for epoch in range(self.config.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_loss = self.run_epoch(sess, train_buffer, dev_buffer)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                if saver:
                    print("New best dev loss! Saving model in ../data/weights/predictor.weights")
                    saver.save(sess, '../data/weights/predictor.weights')
            print()
            
            
    def count_trainable_params(self):
        shapes = [functools.reduce(lambda x,y: x*y,variable.get_shape()) for variable in tf.trainable_variables()]
        return functools.reduce(lambda x,y: x+y, shapes)
