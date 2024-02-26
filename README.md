# Actor-Critic Networks with Analogue Memristors Mimicking Reward-Based Learning
This repository reproduces the experiments of the paper titled: *Actor-Critic Networks with Analogue Memristors Mimicking Reward-Based Learning*.

## Abstract
Advancements in memristive devices have given rise to a new generation of specialized hardware for bio-inspired computing. However, the majority of these implementations only draw partial inspiration from the architecture and functionalities of the mammalian brain. Moreover, the use of memristive hardware is typically restricted to specific elements within the learning algorithm, leaving computationally expensive operations to be executed in software. Here, we demonstrate actor-critic temporal difference (TD) learning on analogue memristors, mirroring the principles of reward-based learning in a neural network architecture similar to the one found in biology. Within the learning algorithm, memristors are used as multi-purpose elements: They act as synaptic weights that are trained online, they calculate the weight updates directly in hardware, and they compute the actions for navigating through the environment. Thanks to this, weight training can take place entirely in-memory, eliminating the need for data movement, enhancing processing speed. Our proposed learning scheme possesses self-correction capabilities that effectively counteract noise during the weight update process, making it a promising alternative to traditional error mitigation schemes. We test our framework on two classic navigation tasks - the T-maze and the Morris water-maze - using analogue valence change memory (VCM) memristors. Our approach represents a first step towards fully in-memory, online and error-resilient neuromorphic computing engines based on bio-inspired learning schemes.

## Instructions 
```bash
# Clone the repository
$ git clone https://github.com/ztill/TD_learning_on_memristors.git
 
# Extract the results-extract_first.zip in the 'results' folder
# Install and activate base anaconda environment
conda activate baseTD
#Train T-maze by executing the notebook: Grid_Search_Tmaze.ipynb
#Train Morris water-maze running the script: Watermaze_grid_search.py
```

## Experiments in the paper
The plots displayed in the paper can be generated using the scripts:  Grid_Search_Tmaze.ipynb and Plot_Watermaze_Figure_Paper.ipynb

