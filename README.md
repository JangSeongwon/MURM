# MURM
IEEE ACCESS  
--------------------------
*Process after submission: Clear out the codes and sort out*
*Code model update*

```
Check GPU temperature in case  
nvidia-smi -q 
device = torch.device('cuda:0') 

import rlkit.torch.pytorch_util as ptu  
ptu.set_gpu_mode(True)  

RAM check  
sar -r 1
```
  
```  
sudo mount /dev/sda2 /media/skc/
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

Anaconda env = conda create -n murm python=3.8
```   
Need at least 240GB space / 200GB=demos  
For Computer2: sudo mount /dev/sda2 /media/skc/  

```
Using 2TB HDD (+ 1TB SSD)
path: /media/jang/jang/0ubuntu/ 
      demos_dataset, image_dataset , images , presampled_goals, Video_Check , Vae_Model  
      RandomBox, SingleView, Wall, z_Running_test  
                        demos
      800 / 800 / 400 
        
      2100 episodes of images(900,600,600) = 577.5K, same for 64 images   
  
```

--------------------------
# Environment Settings

Panda Robot env settings  
  
-Control with end-effector's XYZ coordinate (7DOF using quaternions possible as well)   
-Gripper Control with Distance and Robot finger's Contact  
-RL: Give action with POS  

Env1: Single-view task 
      MURMENV- 9 Boxes as goals, Random initially placed objects + 3 types of shapes + Random color(rgb)  
             - Added Diagnostics + Running for goal part  
    
Env2: Multi-view task  
      MURMENV_m1- Similar to Env1 but global-view is hindered with a wall to goal-boxes (Global camera disadvantage)   
      MURMENV_m2- Randomly positioned goal-box (Active camera disadvantage)   

--------------------------
# Key Points Considered  

***Our prime goal is to compare prior methods(Only single views) with MURM in solving GCRL.***  
***Plus, Multi-views can tackle more complicated tasks that prior methods cannot solve.***  
***Consider Hierarchical Reinforcement Learning for initial state of active-camera and more broader cases as Next Paper***  

1. GCRL components:  
   * Diversity in shapes(cube, rectangular prism1, 2) and colors(Full random) of objects  
   * Fixed 9 goal boxes VS Random 1 goal box   
   * Random initial position(In range) of the object
   * Random Action POS in demos
  
2. Method:  
    *With Demo and VQVAE, train in offline RL   
    *Design reward function computed with global-view first and active-view next   
    *Check whether increading demos have meaningful effect as further work   
    *Use Robot state information as further work   
    *Compare percentage of what not to do + expert dataset size   
            
-------------------------- 

2/17 Analysis   

128*128  
Training time offline = maybe 300 (14 hours), 500 (22 hours)  
Training time online = maybe 250 (24 hours), 100 (9 hours)

64*64
Main computer  
Training time offline = 50 e ( 4.5hour / MURM version)    
Training time online = 50 e ( hours)   
  
computer3  
Training time offline = 100 e ( 4hours )    
Training time online = 50 e ( hours) 
  

1. Collecting demo data with third computer    
=> OKAY  
   
2. Pretrained VQVAE  
=> OKAY -1500 epoches = 6.5days & 2VQVAE model  
   
3. Started training offline RL  
=> OKAY   
       

4. Offline RL framework implementation   
    
    *Primary Concern*  
    - How to use two latent vectors efficiently??  (5120 + 5120 latent space)      
           Images: Simply Concatenate features as in Lookcloser   
           Reward:  
             
    - Point1: Do I have to add noisy dataset??  
    - Point2: How much offline demo data = 1000 as standard     
    - Point3: Add robot state information        
      
     
5. As additional process try images with lower-dimensional size (48x48 as lowest possibility)   
  
6. Edited rollout functions and goal sampling part and diagnostics   
      -> Edited murm in codes + online tuning codes  
      -> Edited murm_env_m3 edit + Final goal image changing env    
      -> Edited for 64 size images settings  
      -> Now try and check...   

--------------------------

# Next Process for Paper 2  

*More updated model with HRL  
  
In Low-level policy: Active-view and Global-view can be considered more suitably and efficiently      
-Finding and detection of the object: Considering a more variety of initial state in terms of active-view camera    
-Pick & Placing in Multi-view task: More organized reward function and structure of MURM  
***Furthermore, Hierarchical multi-scale latent maps can deal with high-resolution images needed necessarily for complicated tasks.***  

--------------------------
Possible implementation of VQVAE2:  
      *Parameters of VQVAE2:*  
      Beta = 0.25  
      weight decay = 0  
      latent_loss_weight = 0.25  
      batch = 128  
