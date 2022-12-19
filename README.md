# MURM

--------------------------

During Training RL,
nvidia-smi -q (Check GPU temperature)

--------------------------

## Installation

Used : PyBullet, Pytorch

```
python = 3.8 version
conda env create -f setup/murm1.yml
conda activate murm1

```

--------------------------

Panda Robot env settings  
  
-Control with end-effector's XYZ coordinate  
-Gripper Control with Distance and Robot finger's Contact  
-RL: Give action with POS  

--------------------------
Env1: Single-view task with detail needed to solve
    
Env2: Multi-view task (100%)

--------------------------

Diversity of objects in task since Offline RL  
  
For online RL, only adding Exploration  part as a theme  
  
Replay Buffer (CCVAE참조로 z bar)  

Action diversity => Gripper Turning possible? 
문제는 Demo dataset 크기 문제로 보임 -> Collection takes too long 

--------------------------

12/14 Analysis  
1. Pretrain VAE or not  
    -> in awac_rig.py
    VAE Training: from rlkit.torch.grill.common import train_vae
2. Save in .pt (python file)


--------------------------

