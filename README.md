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

Envs: 
      MURMENV- 9 Boxes as goals, Holding the object at start + cube + Random color(rgb)   
      MURMENV_v2- Randomly positioned goal-box   
      MURMENV_v3- Picking up random shaped and randomly positioned object   

--------------------------
# Key Points Considered  

***Our prime goal is to compare prior methods(Only single views) with MURM in solving GCRL.***  
***Plus, Multi-views can tackle more complicated tasks that prior methods cannot solve.***  

1. GCRL components:  
   * Diversity in shapes(cube, rectangular prism, tetris shapes) and colors of objects  
   * Fixed 9 goal boxes VS Random 1 goal box   
   * Random Action POS in demos
  
2. Method:  
    *With Demo and VQVAE, train in offline RL     
    *Check whether increading demos, noisy demos have meaningful effect as further work   
    *Use Robot state information as further work   
            
-------------------------- 

3/30 Analysis       

48*48 version...  
Main computer  
Training time offline = 100 e ( 1 hour / MURM version)    
Training time online = 50 e ( ? hours)   
    
computer2  
Training time offline = 100 e ( 2.5 hours )     
Training time online = 50 e ( ? hours) 

computer3  
Get images and demos
          
*GCRL implementation complete      
* Added Q-functions modification and Dropout modification  

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

128*128  
Training time offline = maybe 300 (14 hours), 500 (22 hours)  
Training time online = maybe 250 (24 hours), 100 (9 hours)  
