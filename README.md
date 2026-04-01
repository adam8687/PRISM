# PRISM
Train imitation learning policy that contains unsafe actions /w dataset (get from Xu) 
Get the demonstrations 
ROBO MME ? 
Collect dataset of unsafe actions 
Next Week: train the IL policy 

Include Paper in which you compare how many safe vs unsafe actions occur based off; if you modify the data of a policy and the safety as a result based off unsafe vs safe actions 
RL is unfeasible currently; seeing unsafe vs safe data 
Next Steps: 
Establish simulator being worked with
Record robot observes observation with stepping through the actions or tasks
200 (n) step rollout within environment with 0 action of staying in place
Create python project, set up simulator, step through chosen environment, control the robot through demonstrations, collect safe and unsafe  demonstrations and save demonstration in a file; 50 safe demonstrations in HDF5 and 25 unsafe demonstrations in another file
Have the bc policy step through the environment and qualitatively measure; make ROBO Suite tasks
ROBO Suite

Make repo, make environment, make video for demonstrations /w rollout 

Use robo suite data collection script to collect demonstrations; take the demonstrations and put in HDF5 file; use that to train the policy

Collect observations

pip install mujoco
pip install robosuite


