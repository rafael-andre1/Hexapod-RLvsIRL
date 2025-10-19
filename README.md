# Reinforcement Learning for Hexapod Robot

This project focuses on developing specialized AI workflows to enable a hexapod robot (Mantis) to move autonomously using Reinforcement Learning (RL). The objective was to build a stable and adaptive locomotion system capable of handling different terrains, including flat and uneven surfaces, as well as stair-climbing tasks.

The final system successfully demonstrated:

* Stable standing and posture control
* Forward walking with minimal body oscillation
* Adaptive control for rough terrains and stair climbing

Additionally, the project explored imitation-based techniques such as Inverse Reinforcement Learning (IRL) for performance comparison against pure RL methods.

## System Setup

**Simulator:** Webots
**Robot Model:** Mantis Hexapod

**Sensors:**

* Inertial Measurement Unit (IMU) for orientation and acceleration
* Position sensors for each joint
* Feet contact sensors
* Center of mass localization

**Actuators:** Motorized leg joints
**Environment:** Flat checkerboard terrain and custom uneven terrain environments


## Algorithm and Approach

The control policy was trained using **Proximal Policy Optimization (PPO)** due to its stability in continuous action spaces. The agent received reward signals based on its ability to:

1. Stand upright and maintain a stable center of mass
2. Move forward without excessive wobbling or falling

Randomized initial joint positions were used during training to promote robust policy learning.

For comparison, **Expert Learning** approaches were implemented using Webots’ built-in expert demonstrations.

## Experiments and Performance Metrics

### Standing Up

**Metric:** Variation of center of mass within a defined threshold
**Success Criterion:** Robot reaches and maintains stable upright posture

### Walking

**Metrics:**

* Forward distance traveled per fixed time
* Stability score based on IMU data (orientation and acceleration variance)
  **Success Criterion:** Robot walks a defined distance without falling or excessive body oscillation


## Results

The trained agent achieved stable locomotion across flat terrain and successfully performed controlled stair-climbing maneuvers under simulation. Comparative analysis showed that imitation-based policies accelerated convergence, while PPO provided superior long-term stability and adaptability.


## Known Issues and Documentation

A detailed discussion of encountered issues, challenges, and troubleshooting steps is provided in the accompanying **issues.pdf** file.


## Tutorial: 

1. Install python 3.9 in your computer

2. Set your interpreter to use python 3.9

3. Create your virtual environment by using your version of: 
   "/usr/local/webots/lib/controller/python"

4. Install requirements on your virtual environment/venv:
   "pip install -r requirements.txt"

5. Run `train.py`, answer the input questions and enjoy!



## Disclaimers:

 - Should be compatible with both Linux and Windows, but all of training and testing was carried out in Windows. Simply change line 51 in the `heaxpod_env.py`, modifying 'webots_cmd' to:
   "webots_cmd = os.environ.get("WEBOTS_CMD", "webots")"

 - Webots must be closed whenever you attempt to run train.py, as it will count as an instance of a socket and crash the initialization process.

 - DO NOT SAVE the Webots world whenever you finish training, as the reset function simply restarts the world. In case you accidentally overwrite the predefined initial position, there are instructions in the mantis.py file on how to set it up, as per empirically defined.

 - You can stop training at any time. Interruptions will result in graceful stops, creating logs and saving the model with no overwrites.
