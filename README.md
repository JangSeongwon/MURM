# MURM

--------------------------

During Training RL,
nvidia-smi -q (Check GPU temperature)

--------------------------

## Installation

Used : PyBullet, Pytorch

```
python = 3.8 version

```

--------------------------
# Environment Settings

Panda Robot env settings  
  
-Control with end-effector's XYZ coordinate  
-Gripper Control with Distance and Robot finger's Contact  
-RL: Give action with POS  

  
Env1: Single-view task with details needed to solved
      - 9 Boxes as goals 
    
Env2: Multi-view task  
      MURMENV- Similar to Env1 but global-view is hindered with a wall to goal-boxes (Global camera disadvantage)   
      MURMENV1- Randomly positioned goal-box (Active camera disadvantage)   

--------------------------
# Key Points Considered  

***Our prime goal is to compare prior methods(Only single views) with MURM in solving GCRL.***  
***Plus, Multi-views can tackle more complicated tasks that prior methods cannot solve.***  
***Furthermore, Hierarchical multi-scale latent maps can deal with high-resolution images needed necessarily for complicated tasks.***  
***Consider Hierarchical Reinforcement Learning for initial state of active-camera and more broader cases as Next Paper***  

1. GCRL components:  
   * Diversity in shapes(cube, rectangular prism) and colors(Full random) of objects  
   * Fixed 9 goal boxes VS Random 1 goal box   
   * Random initial position(In range) of the object
   * Random Action velocity in demos

2. Online RL will add Exploration to finetune MURM thus increase final success rate result in paper   
  
3. Replay Buffer (CCVAE참조: Use z bar as representation)  

4. Action diversity => Only 4-DOF 

5. Method:  
    *Sample active-view goal image using a conditional decoder with 9 boxes images as condition   
    *Do exactly same with global-view image  
    *Design reward function computed with global-view first and active-view next without condition thought in MURM paper  
      
  
-> 문제는 Demo dataset 크기 문제로 보임 -> Collection takes too long 
+Deal with this by changing task with less action space needed    
=> Now, single episode takes about 45seconds(185 timesteps)
=> For at least 50K transitions, 300 episodes == About 4hours 

--------------------------

12/22 Analysis  
1. Pretrain VAE or not  
    -> in awac_rig.py
    VAE Training: from rlkit.torch.grill.common import train_vae, Model can be Saved in .pt (python file)  
 
2. Start VAE training and img confirmation process with VQVAE Trainer 2

--------------------------

# Next Process for Paper 2  

*More updated model with HRL  

In Low-level policy: Active-view and Global-view can be considered more suitably and efficiently      
-Finding and detection of the object: Considering a more variety of initial state in terms of active-view camera    
-Pick & Placing in Multi-view task: More organized reward function and structure of MURM  


--------------------------

