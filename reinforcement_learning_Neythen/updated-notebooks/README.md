## Reinforcement learning course materials

## Running code locally 
* Using python script
```
conda activate l2dVE
python session3-bioreactor.py
```

* Using jupyter notebook
``` 
conda activate l2dVE && jupyter notebook --browser=firefox
jupyter nbconvert --to python session3-bioreactor.ipynb #to convert notebook to python script
```


## `session1-tabular.ipynb`
- Introduction to reinforcement learning
- Background
    - Ancient history
    - Modern history
    - Useful applications
    - Recommended reading
- Case study
    - Temporal difference agents
        - SARSA 
        - Q-learning
    - Monte Carlo agents
- End-of-chapter exercises

## `session2-deep.ipynb`
- Deep reinforcement learning: Introduction
- Network construction
- Memory
- Agent
- Algorithm specification
- Case study
- End-of-chapter exercises

## `session3-bioreactor.ipynb` -- Real-world example
> Treloar, Neythen J., Alex JH Fedorec, Brian Ingalls, and Chris P. Barnes. "Deep reinforcement learning for the control of microbial co-cultures in bioreactors." PLoS computational biology 16, no. 4 (2020): e1007783. [DOI](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007783) [google-citations](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=17698721817212738220)

1. Setting up DQN agent
```
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

2. Setting up Bioreactor Environment
```
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

3. Assigments with `DQN_agent(BioreactorEnv, n_states, n_actions)`




