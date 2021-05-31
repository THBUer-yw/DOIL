# Dense connections based Off-policy adversarial Imitation Learning

PyTorch implementation of Dense connections based Off-policy adversarial Imitation Learning (DOIL).

In DOIL, we use the [TD3](https://arxiv.org/pdf/1802.09477.pdf) algorithm to train the imitation policy. In addition, [dense connections](https://arxiv.org/pdf/1608.06993.pdf) are integrated into the actor network and the critic network of DOIL. Both the TD3 algorithm and dense connections are beneficial for improving the sample efficiency of [GAIL](https://arxiv.org/pdf/1606.03476.pdf).

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch 1.4](https://github.com/pytorch/pytorch) and Python 3.7. 

### Expert data
We use the official TD3 code of [D2RL](https://github.com/pairlab/d2rl) to train the agent. And then, the trained agent is used to generate expert trajectories. The expert data of Ant-v2, BipedalWalker-v3, HalfCheetah-v2, Hopper-v2, Reacher-v2 and Walker2d-v2 is available at this [Google drive site](https://drive.google.com/drive/u/0/folders/1hlZlOqeQjim8puE0zGxPg3curmhCHGwx).  

### Usage
The ablation experiments can be reproduced by running:
```
./run_ablation.sh
```
The main experiments for DOIL can be reproduced by running:
```
./run_experiments.sh
```

We can also run experiments when reward types vary and using only states transitions by changing the arguments **reward_type** and **states_only**, respectively. 

If the argument **wdail** is set ture, then WGAN is used to train the discriminator, just try it!
### Results
