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
Need at least 240GB space  
For Computer2: sudo mount /dev/sda2 /media/skc/  

```
Using 2TB HDD (+ 1TB SSD)
path: /media/jang/jang/0ubuntu/ 
      demos_dataset, image_dataset , images , presampled_goals, Video_Check , Vae_Model  
      RandomBox, SingleView, Wall, z_Running_test  
                        demos
      800 / 800 / 400 
        
      2100 episodes of images(900,600,600) = 577.5K
  
```

--------------------------
# Environment Settings

Panda Robot env settings  
  
-Control with end-effector's XYZ coordinate (7DOF using quaternions possible as well)   
-Gripper Control with Distance and Robot finger's Contact  
-RL: Give action with POS  

  
Env1: Single-view task 
      MURMENV- 9 Boxes as goals, Random initially placed objects + 3 types of shapes + Random color(rgb)  
    
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

2. Online Tuning needed - Add Exploration to finetune MURM thus increase final success rate   
  
3. Replay Buffer

4. Action diversity => Only 4-DOF 

5. Method:  
    *With Demo and VQVAE, train in offline RL   
    *Design reward function computed with global-view first and active-view next  
    *Use Robot state information as further work  
    *Compare percentage of what not to do + expert dataset size  
            
-------------------------- 

1/21 Analysis  

1. Collecting demo data with three computers    
=> OKAY 

2. Need to pretrain VQVAE. 

=> OKAY - Perhaps 1500 epoches = 6.5days & 2VQVAE model Sooner stop and use   
=> VQVAE with 2100 episodes  
=> Need to check Offline RL training results

3. Use latent space specification method for goal sampling active camera goals  
      - Condition active image goal to end  
      - Global as wide  
      - Reward global achieve as the same time active 
      => Problem: Taking too long to sample a goal + Not working
      
4. Need to think of active-disadv task 
    - Random Goal box as solution

5. Offline RL framework implementation   
    - PtP Analysis needed to utilize   
    - IQL algorithm
    
    *Primary Concern*  
    - How to use two latent vectors efficiently??    
           Images: Simply Concatenate features as in Lookcloser   
           Reward:  
             
    - Point1: Do I have to add noisy dataset??  
    - Point2: How much offline demo data??   
    - Point3: Add robot state information  
       
        
6. *If* success rate is satisfactory, Go for online fine-tuning as well     
   *If not* = Need to also implement online fine-tuning and check the results again   
            
*The conclusion is first to think of way how to use online fine-tuning in my case*   
 We can first expect task similarity & Use GCB theory      
    
7. As additional process try images with lower-dimensional size (64x64 or 48x48 as lowest possibility)  
  
  
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
