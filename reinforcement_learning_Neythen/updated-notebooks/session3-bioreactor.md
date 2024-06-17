# `session3-bioreactor.ipynb`: "Deep Q network for bioreactor optimisation"
* Author(s) for paper and code: Neythen J. Treloar
* Author(s) for educational material: Saba Ferdous, Ed Lowther, and  Miguel Xochicale

## Questions
* How can an agent learn Chemostat environment that can handle an arbitrary number of bacterial strains?

## Objectives
* Learn how to use Deep Q network for Chemostat environments

## Prerequisites
Session 1: Reinforcement learning with tabular value functions
Session 2: Deep reinforcement learning

## Introduction 
In this notebook, we demonstrate the key parts of a DQN agent and then apply that to the maximisation of the product output of a microbial co-culture growing in a bioreactor. 

For `DQN_agent()` The configuration variables are similar to those from the session two notebook, with one exception - we introduce `TAU` to enable us to perform soft updates on the parameters of the $ Q_{target} $ network, so that they shift towards the $Q$ network parameters incrementally rather than duplicate them at a single time step. We're also changing the effect of the `UPDATE_EVERY` variable - this now becomes the frequency with which we perform both the gradient descent step on the $Q$ network parameters and the soft update of the $Q_{target}$ parameters. 

* References 
> Treloar, Neythen J., Alex JH Fedorec, Brian Ingalls, and Chris P. Barnes. "Deep reinforcement learning for the control of microbial co-cultures in bioreactors." PLoS computational biology 16, no. 4 (2020): e1007783. [DOI](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007783) [google-citations](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=17698721817212738220)



## 1. Setting up DQN agent
```python
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, layer1_size=64, layer2_size=64):
    def forward(self, x):

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size):
    def add(self,state, action, reward, next_state,done):
    def sample(self):

class DQN_agent():
    Agent that interacts with and learns from an environment using artificial neural networks 
    to approximate its state-action value function
       def __init__(self, 
                  env, 
              state_size, 
              action_size,
              BUFFER_SIZE = int(1e5),
              BATCH_SIZE = 64,
              GAMMA = 0.99,
              TAU = 1e-3,
              LR = 5e-4,
              UPDATE_EVERY = 4):
         self.q_network = QNetwork
         self.q_network_target = QNetwork
         self.optimizer = optim.Adam
         self.memory = ReplayBuffer
      def get_explore_rate(self, episode, decay):
      def policy(self, state, epsilon=0):
      def update_target(self, model, target_model):
      def update_Q(self, experiences):
      def train(self, n_episodes=200, max_t=1000, decay=None, verbose=True):
```

## 2. Setting up Bioreactor Environment
```python
class BioreactorEnv():
   def __init__(self, 
                    xdot, 
                    reward_func, 
                    sampling_time, 
                    num_controlled_species, 
                    initial_x, max_t, 
                    n_states = 10, 
                    n_actions = 2, 
                    continuous_s = False):
    def step(self, action):
    def get_state(self):
    def action_to_u(self,action):
    def pop_to_state(self, N):
    def reset(self, initial_x = None):

def monod(C, C0, umax, Km, Km0):
def xdot_product(x, t, u):
        R = monod(C, C0, umax, Km, Km0)
def reward_f(x):
```



## 3. Results

![fig](fig-actions.png)  
**Fig** Actions

![fig](fig-population_cells.png)  
**Fig** Population cells 

![fig](fig-return_explore_rate.png)  
**Fig** Return and explore rate


## 4. Assignments 
1. Change intervals of reward_function using "[N1, N2] = [20, 30] × 10^9 cells L−1." to add your conclusions on how the performance of the agent improves or worsened and the explore rate decreases during training. Plot results with returns and explore_rates).
2. How `N_1` and `N_2` from env.xs maintain optimal level for product production (plot results with plt.plot(np.arange(len(env.xs)) `sampling_time`, [x[0] for x in env.xs], label = '$N_1$')).
3. How RL agent is affected if you use "infrequent sampling" (see (Treloar et al, 2020) for further details on infrequent sampling )?

