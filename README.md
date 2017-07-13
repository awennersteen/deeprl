# Using reinforcement learning to train an agent to get a goal

This project is based on:

*Part 1:* https://medium.com/@harvitronix/using-reinforcement-learning-in-python-to-teach-a-virtual-car-to-avoid-obstacles-6e782cc7d4c6

and

https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

The main files are `qlearn.py` where it is defined the ConvNet and the training process.

The game/scene is defined in the file `flat_game/carmunk.py`, you can modify the rules of the game there.

## Installing

These instructions are for a fresh Ubuntu 16.04 box. 

### Basics

Recent Ubuntu releases come with python3 installed. I use pip3 for installing dependencies so install that with `sudo apt install python3-pip`. Install git if you don't already have it with `sudo apt install git`.

Then clone this repo. 

### Python dependencies

`pip3 install numpy keras h5py`

That should install a slew of other libraries you need as well.

### Install Pygame

Install Pygame's dependencies with:

`sudo apt install mercurial libfreetype6-dev libsdl-dev libsdl-image1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libsdl-mixer1.2-dev libswscale-dev libjpeg-dev`

Then install Pygame itself:

`pip3 install hg+http://bitbucket.org/pygame/pygame`

### Install Pymunk

This is the physics engine used by the simulation. It just went through a pretty significant rewrite (v5) so you need to grab the older v4 version. v4 is written for Python 2 so there are a couple extra steps.

Go back to your home or downloads and get Pymunk 4:

`wget https://github.com/viblo/pymunk/archive/pymunk-4.0.0.tar.gz`

Unpack it:

`tar zxvf pymunk-4.0.0.tar.gz`

Update from Python 2 to 3:

`cd pymunk-pymukn-4.0.0/pymunk`

`2to3 -w *.py`

Install it:

`cd ..`
`python3 setup.py install`

Now go back to where you cloned `deeprl` and make sure everything worked with a quick `python3 qlearn.py -m "Run"`. If you see a screen come up with a little dot flying around the screen, you're ready to go!

## Training

First, you need to train a model. This will save weights to the `model.h5` file and its structure definition on `model.json` .  You can train the model by running:

`python3 qlearn.py -m "Train"`

It can take anywhere from an hour to 36 hours to train a model, depending on the complexity of the network and the size of your sample. 

## Playing

`python3 qlearn.py -m "Run"`

That's all there is to it.

## More references

- Playing Atari with Deep Reinforcement Learning - http://arxiv.org/pdf/1312.5602.pdf
- Deep learning to play Atari games: https://github.com/spragunr/deep_q_rl
- Another deep learning project for video games: https://github.com/asrivat1/DeepLearningVideoGames
- A great tutorial on reinforcement learning that a lot of my project is based on: http://outlace.com/Reinforcement-Learning-Part-3/
