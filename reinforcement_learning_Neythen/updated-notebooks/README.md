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
- DQN agent (e.g, QNetwork, ReplayBuffer, DQN_agent)
- Bioreactor Environment (e.g, monod, xdot_product, reward_f)
- Assigments with `DQN_agent(BioreactorEnv, n_states, n_actions)`

