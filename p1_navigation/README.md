# Navigation project

The root of this folder contains

- `Report.pdf`: provides a description of the environment including the following:
	- A despription of the learning algorithm.
	- A plot of the rewards per episode to illustrate that the agent is improving performance over time.
	- Ideas for future work.
- `Navigation.ipynb`: A jupyter notebook that contains definitions for the agent and the Q-network and memory that the agent utilises to calculate output and store experiences. The notebook also contains code that either trains the agent in the Unity environment or loads a set of pre-trained weights.
- `model.pth`: A PyTorch serialization of learned model parameters from a training session. 

## Contents

1. [Project Details](#project-details)
2. [Getting Started](#getting-started)
3. [Instructions](#instructions)

## Project Details

The invironment is a large, square world populated by blue and yellow bananas. 
The goal of the agent navigating this environment is to collect as many yellow bananas as possible while avoiding the blue bananas.
The agent navigating the environment has four discrete actions available to it, corresponding to:

- **`0`** - move forward
- **`1`** - move backward
- **`2`** - turn left
- **`3`** - turn right

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of object around agent's forward direction.
The task is episodic. The task is considered solved when the agent achieves an average score of +13.0 for 100 consecutive episodes.

## Getting Started

Install pytorch and unity ML environment

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file.  

## Instructions

Run all the code cells in the notebook.

When it comes to showing the agent perform in an episode, there are two choices available:
1. Training the agent to learn the model weights for performing the task optimally. 
This option takes the longest time. The time taken depends on the specific computer and whether it has a CUDA-graphics processor available or not.

2. Loading a set of pretrained weights into the model. This way training does not have to be done. One of the cell blocks loads the pre-trained weights saved in the file `model.pth`. 


