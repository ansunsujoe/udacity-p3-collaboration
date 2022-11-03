# udacity-p3-collaboration

In this project, we will be training a pair of agents (via a DDPG) to play a game of friendly tennis.

## Project Details

This task is single-agent and episodic and has a continuous action space. The high-level goal is for two agents to keep the ball in play for as long as possible in the game of tennis.

**State Space**: contains 8 dimensions corresponding to the position and velocity of the ball and tennis racket. This is local with respect to each agent.

**Action Space**: A continuous vector with 2 dimension (one for forward/backward and one for jumping).

**Rewards**: We will get a reward of +0.1 for each time an agent hits the ball over the net correctly, but will get a reward of -0.01 for each time an agent hits the ball in the ground or out of bounds.

**Goal**: In order to solve the environment, our agent must get an average score of +0.5 over 100 consecutive episodes.

## Getting Started

**Step 1:** Clone the Udacity deep-reinforcement-learning project.

```
git clone https://github.com/udacity/deep-reinforcement-learning.git
```

**Step 2:** Copy the folder called "python" in the Udacity deep-reinforcement-learning project into the root directory of this project. This contains the code required to run a Unity environment.

**Step 3**: Open the Jupyter notebook "Tennis.ipynb" and run the first cell in Step 1 in order to install all the necessary Python packages and the unityagents package.

## Instructions

Here are the steps that are required in order for you to be able to train a DDPG agent on the Tennis environment.

**Step 1:** Run all the cells in sections 1-7 of Tennis.ipynb. This will set up all the classes and functions needed for training a DDPG.

**Step 2:** At the beginning of section 8 of the notebook, replace the file path of the Tennis Environment with the file path for your executable.

```
# Create environment
env = TennisEnvironment(fp="/your-file-path")
```

**Step 3:** Tweak any hyperparameters that you want in the initialization of the agent or the calling of the train_ddpg method in Section 8.

**Step 4:** Run the code cell in section 8 and watch your DDPG being trained!
