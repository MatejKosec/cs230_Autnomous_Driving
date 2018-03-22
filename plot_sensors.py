from matplotlib import pyplot as plt
import numpy as np


def plot(sensor_inputs,steering_direction,img=None,savefile=None):
    #print('Sensor inputs: ', sensor_inputs)
    sensor_inputs *= 200 #outputs divide distance by 200
    plt.figure(figsize=(12,6))
    axe = plt.subplot(121, projection = 'polar')
    #axe=fig.gca(polar=True)
    plt.title('Distance to edge of road as measured by sensors [m]')
    axe.set_theta_zero_location('N')
    rlim = 30
    #Plot the sensor inputs
    sensor_inputs = np.minimum(sensor_inputs,rlim)
    axe.set_rlim(0,rlim)
    axe.set_yticks(range(0, rlim, 10))
    sensor_angles=np.linspace(-np.pi/2.0,np.pi/2.0,len(sensor_inputs))*-1
    bars = axe.bar(sensor_angles, sensor_inputs, width=5.0*np.pi/180.0, bottom=0.0)
    # Use custom colors and opacity
    for r, bar in zip(sensor_inputs, bars):
        bar.set_facecolor(plt.cm.jet_r(r/rlim/2))
        bar.set_alpha(0.9)
    
    #Plot the car
    theta = [np.deg2rad(i) for i in [-180,-150,-30,0,30,150,180]]
    r1 = abs(2.0/np.cos(theta))
    r2 = [0]*len(r1)
    axe.plot(theta, r1, color='k')
    axe.fill_between(theta, r1, r2, alpha=0.2,color='k')
    
    #Plot the steering direction
    print('Sterring direction: ', steering_direction)
    theta= [0,np.deg2rad(steering_direction*90)]
    r  = [0,rlim*0.8]
    axe.plot(theta,r,c='k',lw='4')
    
    axe.plot(sensor_angles, sensor_inputs, color='grey')
    
    if type(img) != type(None): 
        ax2 = plt.subplot(122)
        plt.title('Corresponding visual observation [64x64 px]', y=1.045)
        #plt.plot(range(10), range(10))
        plt.imshow(np.reshape(img, [64,64,3]), origin='lower')
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklines(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax2.get_yticklines(), visible=False)
    if type(savefile) != type(None):
        plt.savefig(savefile, dpi=250, bbox_inches='tight')
    else:
        plt.show()

if __name__ =="__main__":
    s = """
    0.332652   0.1069225  0.076417   0.061924   0.055318   0.052476
 0.0510695  0.04989655 0.0490901  0.0483087  0.0475512  0.0468169
 0.04582625 0.04474455 0.04285135 0.03952915 0.0350934  0.0305466
 0.02243225"""
    s2 = s.replace('\n','').strip().replace('  ',' ').replace('  ',' ').replace('  ',' ').split(' ')
    sensor_inputs = np.array(s2, dtype=np.float32)
    steering_direction = -0.7
    img = np.random.rand(64,64,3)
    plot(sensor_inputs,steering_direction, img, '../experiments/test.png')