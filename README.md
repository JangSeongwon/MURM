# MURM

--------------------------

```
Check GPU temperature in case  
nvidia-smi -q 

RAM check  
sar -r 1
```
--------------------------

## Installation Needed

Used : PyBullet, Pytorch

```
python = 3.8 version
gym==0.12.1
pybullet==2.5.5
pygame==1.9.6
opencv-python

Anaconda env = murm1  
```

Need at least 240GB space

```
Using 2TB HDD (+ 1TB SSD)
path: /media/jang/jang/0ubuntu/ 
      image_dataset , images , presampled_goals, Video_Check , Vae_Model  
      128_100_1, 128_150_5+, 256_50_8+, Running_test  
      275 timesteps per episode 
```

Dataset Size
```
128 image - 150 episodes, 275 timesteps = 41.25K images  
256 image - 50 episodes, 275 timesteps = 13.75K images 

Comparing Standard:
      ImageNet = 1200K
      FFHQ = 70K
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

1/6 Analysis  

1. Collecting demo data with two computers    
=> OKAY 

2. Need to pretrain VAE. 

=> OKAY Perhaps 300 epoches = 1.5days & 6VQVAE model    
=> Getting VQVAE with 300 epoches  
=> Check goal sampling theory with vqvae model i have  

    Parameters of VQVAE2:   
    Beta = 0.25  
    weight decay = 0  
    latent_loss_weight = 0.25  
    batch = 128  

3. Use latent space specification method for goal sampling active camera goals  
      - Condition active image goal to end  
      - Global as wide  
      - Reward global achieve as the same time active 
      
4. Need to think of active-disadv task 
    - RandomGoal box as solution
--------------------------

# Next Process for Paper 2  

*More updated model with HRL  

In Low-level policy: Active-view and Global-view can be considered more suitably and efficiently      
-Finding and detection of the object: Considering a more variety of initial state in terms of active-view camera    
-Pick & Placing in Multi-view task: More organized reward function and structure of MURM  


--------------------------

