import tensorflow as tf
class config():
    #TORCS settings
    vision = False
    throttle = False
    env_name = 'TORCS'

    # output config
    restore_from_ckpt = False
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
   
