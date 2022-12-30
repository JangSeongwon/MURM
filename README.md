# MURM

--------------------------

During Training RL,
nvidia-smi -q (Check GPU temperature in case)

--------------------------

## Installation Needed

Used : PyBullet, Pytorch

```
python = 3.8 version

```

--------------------------
# Environment Settings

Panda Robot env settings  
  
-Control with end-effector's XYZ coordinate (7DOF using quaternions possible as well)   
-Gripper Control with Distance and Robot finger's Contact  
-RL: Give action with POS  

  
Env1: Single-view task 
      - 9 Boxes as goals 
      - Random initially placed objects + 3 types of shapes + Random color(rgb)  
    
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
      
      
-------------------------- 

12/30 Analysis  

1. 문제는 Demo dataset 크기 문제로 보임 -> Collection takes too long 
+Deal with this by changing task with less action space needed    
=> Now, single episode takes about 110seconds(300 timesteps)  
=> For 90K transitions, 300 episodes == About 9hours  

2. Need to pretrain VAE. 
    -128 batches in 1 epoch = 8min, 480seconds
    -Maybe 500 epoches = 67hours, 3days
    - 1000 epoches = 133hours, 6days
    -Save Model in .pt   

--------------------------

# Next Process for Paper 2  

*More updated model with HRL  

In Low-level policy: Active-view and Global-view can be considered more suitably and efficiently      
-Finding and detection of the object: Considering a more variety of initial state in terms of active-view camera    
-Pick & Placing in Multi-view task: More organized reward function and structure of MURM  


--------------------------

