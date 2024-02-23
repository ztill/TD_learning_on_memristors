import pickle
from _context import *
from src.rl import ActorCriticWaterMaze
from src.environments import WaterMaze
from tqdm import tqdm
from src.memristor import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from functools import partial

from scipy.special import softmax
import fnmatch
import json

##--Function Definitions------------
# Function for update of actor weights using multiprocessing 
def update_actor_complete_indiv_memristor(rands_a,i,j,theta,d_t):
    np.random.seed(rands_a)
    # print("a_weight:",i,j, 'Rand. Nr:',np.random.normal(0.5,1),flush=True)
    #Multiple Memristors
    theta, _ = m[mapping_actor[i,j]][0].update_syn_w_emulated(theta, d_t, update_type=RL_settings['update_type'], noise_type=RL_settings['noise_type'], pulse_calc=RL_settings['pulse_calc'])
    #1 Memristor
    #theta, _ = m[3][0].update_syn_w_emulated(theta, d_t, update_type=RL_settings['update_type'], noise_type=RL_settings['noise_type'], pulse_calc=RL_settings['pulse_calc'])
    return theta
    
def get_device_name(word):
    start_string = "_D"
    end_string = "__"

    # Find the indexes of the given letters in the string
    index_start = word.find(start_string)
    index_stop = word.find(end_string)

    if (index_start != -1) or (index_stop != -1):
        # Extract the part of the string starting from the given letter
        result = word[index_start+1:index_stop]

        # print("Part of the string starting with the given letter:", result)
    else:
        print("The given letter was not found in the string.")
    return result
def get_actor_critic_ID(word):
    start_string = "__"
    end_string = ".j"
    # Find the indexes of the given letters in the string
    index_start = word.find(start_string)
    index_stop = word.find(end_string)

    if (index_start != -1) or (index_stop != -1):
        # Extract the part of the string starting from the given letter
        result = word[index_start+2:index_stop]

        # print("Part of the string starting with the given letter:", result)
    else:
        print("The given letter was not found in the string.")
    return result

######### Create Memristor Emulations ###########
n_states=27 #number of memristors
actions=8
critics=1

#init class for each memristor
memristors=[[Memristor()] for i in range(n_states)] #memristors[state][0=Critic or 1=Actor][if actor which of two actions] #memristors=[[[],[]]]*n_states does not work!!!!

directory='memristor_data/'

files=os.listdir(directory)
# filenames=fnmatch.filter(files,'*.json')
filenames=sorted(fnmatch.filter(files,'*.json'),key=get_actor_critic_ID)


c_idx, a_idx = 0, 0

#Iterate through files and assign memristors to actors/critics
for i,filename in enumerate(filenames):
    # print('device name:', get_device_name(filename), ', actor/critic ID:', get_actor_critic_ID(filename))
    pathfull = directory+'/'+filename     
    params={
        'N':0, #number of pulses to max conductance
        'ID':"",
        'G0':0,
        'G1':0,

        'alpha_set':0, #parameter nonlinear update fct
        'beta_set':0, #parameter nonlinear update fct
        'alpha_reset':0, #parameter nonlinear update fct
        'beta_reset':0, #parameter nonlinear update fct
        
        'alpha_noise_set':0, 
        'beta_noise_set':0,
        'gamma_noise_set':0,
        'zeta_noise_set':0,
        'alpha_noise_reset':0,
        'beta_noise_reset':0,
        'gamma_noise_reset':0,
        'zeta_noise_reset':0,

        'std_noise_set':0.0,
        'std_noise_reset':0.0,
        'std_noise_set_cycle':np.array([]),
        'std_noise_reset_cycle':np.array([]),

        'sign_counter': 0,
        'update_sign': None, #sign of last update: +1 for positive update, -1 for negative 
        'count_direction': 'up',
        'cycle': None,
        

        # --Data (for Run not needed)--
        # 'Gset':0,
        # 'Greset':0,
        # 'pset':0,
        # 'preset':0,
        # 'Gset_norm':0,
        # 'Greset_norm':0,
    }
    
    file_ID = get_actor_critic_ID(filename) 
    #Assign critic memristors
    if c_idx < critics * n_states:    
        critic_idx = int(file_ID[1:])
        
        if file_ID.startswith('C'):
            #Assign the device ID to the model
            memristors[critic_idx][0].ID = get_device_name(filename)

            with open(pathfull) as json_data:
                data = json.load(json_data)
                #print(data)

            params.update((k, data[k]) for k in params.keys() & data.keys()) #update all values in params with characterization
            memristors[critic_idx][0].update_params(params)
            x=np.arange(memristors[critic_idx][0].N+1)
            # ax1.plot(x,memristors[critic_idx][0].Lin_Exp_Set(x),label=f'SET s{critic_idx}',c=f'C{critic_idx}')
            # ax1.plot(x,memristors[critic_idx][0].Lin_Exp_Reset(x),label=f'RESET s{critic_idx}',c=f'C{critic_idx}')
            c_idx += 1

#Print all G0 and G1 of Memristors
# for i in range(n_states):
#     print(f'Device {i}:','ID:', memristors[i][0].ID)
            

#################################################
####### Settings ###############################
#################################################
m=memristors

save=True

## Savepath
path='/Users/till/Downloads/grid_search/random_mapping_fine/'

# Name of Savefile
name=f'Fine_Grid_Search_50k_step1p5_LR0p07_gamma0p975_multiprocessing_random_mapping_5'


# Environment specs
no_rbf = 11
no_actions = 8
no_states = no_rbf**2


#--Random mapping of actor weights to 27 memristors----
seed=15 #seed with very equal distribution of numbers in 0-26
mapping_actor=np.zeros((no_states,no_actions),dtype=int)
np.random.seed(seed)
for i in range(no_states):
    for j in range(no_actions):
        mapping_actor[i,j] = int(np.random.randint(0,27))

#Multiprocessing
n_processes=4

# Algorithm specs
lrs= [0.07] #list of learning rates for grid search
Ts= [0.075] #list of softmax temperature  for grid search

gammas = [0.975]
step_sizes = [1.5]
std_devs_rbf=[0.75] #0.75 used standard
threshold=0.0025 #threshold of quantization to 1 pulse minimum
num_episodes = [50000]

max_iter = 300
n_seeds=100 #number of seeds per learning setting
seeds=np.arange(n_seeds)
chunk_size = 1 #averaging bin size for number of steps and fraction that reaches reward 

render_seed=False
render=False #render trajectory
plot=False

snapshot = 1000 #interval for snapshots
# mean_interval = 1000 #interval at which prinout
results_interval = 100 #interval at the end where mean of weights is calculated

# Initialization of parameters
RL_settings={
    'update_type': 'model', #options: "ideal", "model", "random cycle"
    'noise_type': 'set reset separate', #options: None, "constant", "set reset separate", "cycle noise"'
    'pulse_calc': 'linear' #options: 'model' (normal update), 'linear' (linear update) 
}

results=[]



######## ######### ######### #########
#### Run/Grid Search #########
######### ######### ######### #########

if __name__ == '__main__':
    print('Filename:',name)
    for num_episode in num_episodes:
        for step_size in step_sizes:
            for gamma in gammas:
                for std_dev_rbf in std_devs_rbf:
                    for lr in lrs:
                        for T in Ts:
                            with ProcessPoolExecutor(n_processes) as executor:
                                learning_rate=lr
                                num_chunks = num_episode // chunk_size

                                print('---step_size',step_size,'std_dev_rbf',std_dev_rbf,'lr',learning_rate,'T',T,'gamma',gamma,'std_dev_rbf',std_dev_rbf,'---')

                                #Initialize numpy arrays for data across seeds
                                final_steps_list=np.zeros(n_seeds)
                                final_rewards_list=np.zeros(n_seeds)

                                environments=[]

                                critic_weights_list=[]
                                actor_weights_list=[]

                                #Snapshots
                                snapshot_list_actor=[]
                                snapshot_list_critic=[]
                                
                                #Mean weights over given last nr of episodes
                                mean_final_critic_weights=[]
                                mean_final_actor_weights=[]

                                steps_list=np.zeros((n_seeds,num_chunks))
                                rewards_list=np.zeros((n_seeds,num_chunks))
                                
                                for seed in seeds:
                                    snap_id = 0
                                    mean_id = 0
                                    print('seed',seed)

                                    #Initialize Pseudo Random Number generation
                                    np.random.seed(seed)
                                    
                                    env = WaterMaze(no_states, st_dev=std_dev_rbf, step_size=step_size, w_scale=1)
                                    net = ActorCriticWaterMaze(no_states, T)
                                    if render_seed:
                                        env.render()

                                    # Initialize weights to 0.5
                                    net.w_a += 0.5
                                    rewards = np.zeros(num_chunks)
                                    lengths = np.zeros(num_chunks)

                                    #Average weights over last few episodes (results_interval)
                                    w_a_mean = np.zeros((results_interval, no_states, no_actions))
                                    w_v_mean = np.zeros((results_interval, no_states))

                                    #snapshots of weights during learning
                                    w_a_snap = np.zeros((num_episode//snapshot, no_states, no_actions))
                                    w_v_snap = np.zeros((num_episode//snapshot, no_states))
                                    

                                    for i in tqdm(range(num_episode)):
                                        env.reset()
                                        total_reward = 0
                                        iteration = 0

                                        while iteration < max_iter and not env.end:
                                            old_state = env._state
                                            # Compute action
                                            a, h_a = net.action(env.get_rbf(old_state))
                                            # Do action
                                            state, reward = env.step(a)
                                            total_reward += reward

                                            new_value = 0 if env.end else net.value(env.get_rbf(state))
                                            old_value = net.value(env.get_rbf(old_state))

                                            delta = reward + gamma * new_value - old_value
                                            
                                            hebbian_actor = np.zeros((no_states, no_actions))
                                            hebbian_actor[:, :] = -np.outer(env.get_rbf(old_state), h_a)
                                            hebbian_actor[:, a] += 1 * env.get_rbf(old_state)

                                            #Cutoff small values (not updated on memristor anyways due to discretization)
                                            delta_w=learning_rate * delta * env.get_rbf(old_state)
                                            delta_theta=learning_rate * delta * hebbian_actor

                                            #Cutoff all updates that are below 1 ulse to the memristor
                                            delta_w[np.abs(delta_w)<threshold]=0
                                            delta_theta[np.abs(delta_theta)<threshold]=0

                                            #------- Options with Cutoff in Updates-------------
                                            nonzero_indices_v = np.nonzero(delta_w)
                                            nonzero_indices_a = np.nonzero(delta_theta)
                                            n_jobs_critic=len(nonzero_indices_v[0])
                                            n_jobs_actor=len(nonzero_indices_a[0])

                                            # Update Critic Weights
                                            if n_jobs_critic>0:
                                                for idx in nonzero_indices_v[0]: #since second dimension is empty
                                                    net.w_v[idx], _ = m[idx%n_states][0].update_syn_w_emulated(net.w_v[idx],delta_w[idx],update_type=RL_settings['update_type'],noise_type=RL_settings['noise_type'], pulse_calc=RL_settings['pulse_calc'])

                                            # Update Actor Weights with Multiprocessing
                                            if n_jobs_actor>0: #check if empty
                                                rands_a=np.random.randint(0,1000,len(nonzero_indices_a[0])) # initialize random numbers to ensure consistent random number generation in multiple processes
                                                iter_a=executor.map(update_actor_complete_indiv_memristor,rands_a,nonzero_indices_a[0],nonzero_indices_a[1],net.w_a[nonzero_indices_a],delta_theta[nonzero_indices_a],chunksize=n_jobs_actor//n_processes+1)
                                                new_a=np.fromiter(iter_a, dtype=float, count=-1)
                                                net.w_a[nonzero_indices_a] = new_a
                                            
                                            #Clip Weights to [0,1]
                                            net.w_v = np.clip(net.w_v, 0, 1)
                                            net.w_a = np.clip(net.w_a, 0, 1)


                                            #Save Snapshots of Weights
                                            if ((i>0) and ((i % snapshot) == 0) and (iteration == 0)):
                                                # print("Snap at Episode", i)
                                                w_a_snap[snap_id] = net.w_a
                                                w_v_snap[snap_id] = net.w_v
                                                snap_id += 1

                                            # Save network weights the last few episodes of "results_interval"
                                            if ((i>=num_episode-results_interval) and (i < num_episode) and (env.end or iteration==max_iter-1)):
                                                w_a_mean[mean_id] = net.w_a
                                                w_v_mean[mean_id] = net.w_v
                                                mean_id += 1
                                            iteration += 1

                                        #Save rewards
                                        rewards[i // chunk_size] += total_reward / chunk_size
                                        lengths[i // chunk_size] += iteration / chunk_size


                                    #Calculate mean of weights over interval "results_interval"
                                    mean_w_a = np.mean(w_a_mean, axis=0)
                                    mean_w_v = np.mean(w_v_mean, axis=0)

                                    #Add results of seed to lists
                                    steps_list[seed]=lengths
                                    rewards_list[seed]=rewards
                                    
                                    final_steps_list[seed]=int(lengths[-1])
                                    final_rewards_list[seed]=round(rewards[-1],3)

                                    environments.append(env)
                                    critic_weights_list.append(net.w_v)
                                    actor_weights_list.append(net.w_a)

                                    # Mean over last episodes
                                    mean_final_critic_weights.append(mean_w_v)
                                    mean_final_actor_weights.append(mean_w_a)
                                    
                                    #Snapshots
                                    snapshot_list_actor.append(w_a_snap)
                                    snapshot_list_critic.append(w_v_snap)


                                    print(f'Final #steps',int(lengths[-1]))
                                    print(f'Final reward fraction',round(rewards[-1],3))

                                    if render:
                                        env.render()

                                    #  --Plot---
                                    if plot:
                                        plot_points = (np.arange(num_chunks) + 1) * chunk_size

                                        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, dpi=400, figsize=(4, 3))

                                        ax1.plot(plot_points, rewards)
                                        ax1.set_title("Fraction of episodes where the goal is \nreached before %d iterations" % 300) # Max iterations is
                                        ax1.set_xlabel("Episode")
                                        ax1.set_ylabel("Fraction")

                                        ax2.plot(plot_points, lengths)
                                        ax2.set_title("Episode length over time")
                                        ax2.set_xlabel("Episode")
                                        ax2.set_ylabel("Episode length")

                                        env.render_actions(net, ax3, dense=True, magnitude=True)

                                        env.render_values(net, ax4, show_rbf=True)

                                        fig.subplots_adjust(wspace=0.5, hspace=0.5)
                                        fig.show()
                                
                                #Calc Mean of Seeds
                                mean_final_steps=np.mean(final_steps_list)
                                mean_final_rewards=np.mean(final_rewards_list)
                                print('Mean #steps in all seeds:',mean_final_steps)
                                print('Mean fraction reward reached in all seeds:',mean_final_rewards)
                                print(len(snapshot_list_critic))
                                # results.append(
                                #         {"LR": lr, "T": T,"step_size":step_size,"std_dev_rbf":std_dev_rbf,"seeds":seeds,"threshold":threshold,"gamma":gamma,"num_episodes":num_episodes,"chunk_size":chunk_size,
                                #             "mean_final_steps":mean_final_steps,"mean_final_rewards":mean_final_rewards,"rewards_list":rewards_list,"steps_list":steps_list,"environments":environments,"critic_weights_list":critic_weights_list,"actor_weights_list":actor_weights_list,
                                #             "snapshot_list_critic":snapshot_list_critic,"snapshot_list_actor":snapshot_list_actor,"final_critic_weights":final_critic_weights,"final_actor_weights":final_actor_weights})
                                results.append(
                                        {"LR": lr, "T": T,"step_size":step_size,"std_dev_rbf":std_dev_rbf,"seeds":seeds,"threshold":threshold,"gamma":gamma,"num_episodes":num_episodes,"chunk_size":chunk_size,
                                            "mean_final_steps":mean_final_steps,"mean_final_rewards":mean_final_rewards,"rewards_list":rewards_list,"steps_list":steps_list,"environments":environments,"critic_weights_list":critic_weights_list,"actor_weights_list":actor_weights_list,
                                            "snapshot_list_critic":snapshot_list_critic,"snapshot_list_actor":snapshot_list_actor,"mean_final_critic_weights":mean_final_critic_weights,"mean_final_actor_weights":mean_final_actor_weights})

                                # Save every time there is a new result  
                                if save:
                                    os.makedirs(path,exist_ok=True) #Create directory if does not exist
                                    fullpath=f"{path}{name}"
                                    results_save = pd.DataFrame.from_dict(results)
                                    results_save.to_pickle(fullpath)