#!/usr/bin/env python
"""
16740-A: HW2
Skill Learning
"""
import argparse, math, os, random, sys, signal, time, scipy.integrate, scipy.optimize, pdb
sys.path.append("utilities")
sys.path.append(os.getcwd())

try:
  from lib import vrep
except:
  print ('"vrep.py" could not be imported. Check for the library file')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import vrepsutil as vu
from RRTTree import RRTTree
from RRTConnect import RRTConnectPlanner
# Constants/hyperparameters - DO NOT MODIFY!
NUM_BASIS = 3
NUM_DIM = 3
TRAJ_HORIZON = 200
COLLISION_TOLERANCE = 0.01
bb_l_door = np.array([0.25,0.005,0.5])
bb_l_door += COLLISION_TOLERANCE
bb_c_door = np.array([0.0,-0.5+0.0025,0.3])

bb_l_handle = np.array([0.05,0.005,0.05])
bb_l_handle += COLLISION_TOLERANCE
bb_c_handle = np.array([0.0,-0.5+0.10025,0.26])

class bcolors:
  """
  ----------------------- DO NOT MODIFY! ---------------
  """
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

def is_in_collision(x, bb_c, bb_l):
  """
  ----------------------- DO NOT MODIFY! ---------------

  Computes whether object with position x is inside the bounding box of 
  the collision object with parameters bb_c and bb_l.

  Parameters
  ----------
  x : Corresponds to the position of the object
  bb_c : a list of box center position [x,y,z]
  bb_l : a list of box dimensions [length, width, height]

  Returns
  ----------
  True if x is within the bounding box, False otherwise
  """
  max_vec = bb_c+bb_l
  min_vec = bb_c-bb_l
  return np.all(np.logical_and(x > min_vec, x < max_vec))


class VRepEnvironment(object):
  """
  ----------------------- DO NOT MODIFY! ---------------

  Wrapper for V-REP environment

  Parameters
  ----------
  task_id : 0 for the first part of the task (reach for door handle),
            1 for the second part (pull door with the handle)

  """
  def __init__(self, task_id):
    self.task_id = task_id
    self.setup_simulation()

  def setup_simulation(self):

    self.client_id = vu.ConnectVREPS()
    vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_oneshot)

    object_desc, joint_desc, self.obstacles = vu.GenConGridObjDesc()

    self.object_handles = vu.GenObjects(self.client_id, object_desc)

    object_desc, joint_desc = vu.GenBasicDoorObjDesc()
    self.object_handles = vu.GenObjects(self.client_id, object_desc)

    self.joint_handles = vu.GenJoints(self.client_id, self.object_handles, joint_desc)
    self.robot_handles = vu.GetRobotHandles(self.client_id)

    self.door_id = 0
    self.doorhandle_id = 1
    pdb.set_trace()

    # Set goal position for reacher task
    self.reacher_goal_position = self.DoorHandlePosition.copy()
    self.reacher_goal_position[1] = (self.DoorPosition[1] + self.DoorHandlePosition[1]) * 0.5

    # Set goal position for pull task
    self.pull_goal_position = self.DoorHandlePosition.copy()
    self.pull_goal_position[0] = 0.2 - 0.05
    self.pull_goal_position[1] = -0.38
    self.pull_goal_position[2] = 0.25

    if self.task_id == 0:      
      vu.SetDesJointPos(self.client_id, self.robot_handles,[0,0.,1.5,0,0,0,0.0,0.0]) 
      vu.StepSim(self.client_id, 5)
    elif self.task_id == 1:      
      self.setRobotPosition(self.reacher_goal_position, dt=20)
    else:
      raise NotImplementedError

  @property
  def ObjState(self):
    return vu.GetObjState(self.client_id, self.object_handles)

  @property
  def RobotState(self):
    return vu.GetRobotState(self.client_id, self.robot_handles)

  @property
  def DoorPosition(self):
    return np.array(self.ObjState["Block"][self.door_id][0:3])

  @property
  def DoorHandlePosition(self):
    return np.array(self.ObjState["Block"][self.doorhandle_id][0:3])

  @property
  def RobotPosition(self):
    return np.array(self.RobotState["EEPose"][0:3])

  @property
  def GoalPosition(self):
    return self.reacher_goal_position if self.task_id == 0 else self.pull_goal_position

  def computeReward(self):
    return self.computeReacherReward() if self.task_id == 0 else self.computePullReward()

  def computeReacherReward(self):
    robot_pos = self.RobotPosition

    doorhandle_pos = self.DoorHandlePosition
    door_pos = self.DoorPosition
    goal_pos = self.reacher_goal_position

    # Attraction reward
    dist_from_goal = np.sqrt(np.sum((robot_pos - goal_pos)**2))
    attractor_reward = 1.0 if dist_from_goal < 0.03 else 0.

    # Repulsive reward
    repulsive_reward_door = -1 if is_in_collision(robot_pos, bb_c_door, bb_l_door) else 0
    repulsive_reward_handle = -1 if is_in_collision(robot_pos, bb_c_handle, bb_l_handle) else 0
    repulsive_reward_floor = -1 if robot_pos[2] < 0.1 else 0

    repulsive_reward = repulsive_reward_door + repulsive_reward_handle + repulsive_reward_floor

    return attractor_reward + repulsive_reward

  def computePullReward(self):
    door_pos = self.DoorPosition
    attractor_reward = 1 if door_pos[1] > -0.4 else 0
    return attractor_reward

  def signal_handler(self, sig, frame):
    print('Ctrl-C pressed!')
    self.close()
    sys.exit(0)

  def executeTrajectories(self, trajectories, evaluate=False):
    # Get number of trajs
    n_trajs = trajectories.shape[0]    
    return_trajs = np.zeros((n_trajs, 1))
    for traj_id in range(n_trajs):
      return_trajs[traj_id] = self.executeTrajectory(trajectories[traj_id, :])
      print(bcolors.UNDERLINE+'Trajectory '+str(traj_id)+' obtained return '+str(return_trajs[traj_id])+bcolors.ENDC)
      if not evaluate:
        # Reset environment
        self.reset()

    return return_trajs

  def executeTrajectory(self, trajectory):
    return_traj = 0.
    for pose in trajectory:
      reward, doorhandle_pose = self.step(pose)
      return_traj += reward

    return return_traj

  def step(self, robot_pose):
    self.setRobotPosition(robot_pose)
    reward = self.computeReward()
    doorhandle_pose = self.DoorHandlePosition
    return reward, doorhandle_pose

  def setRobotPosition(self, robot_pos, dt=1):
    offset_robot_pos = robot_pos
    offset_robot_pos[1] -= 0.5
    vu.SetDesJointPos(self.client_id, self.robot_handles, list(offset_robot_pos) + [0,0,0,0.0,0.0])
    vu.StepSim(self.client_id, dt)

  def close(self):
    """
    Exit cleanly, stop V-REP execution
    """
    vrep.simxStopSimulation(self.client_id, vrep.simx_opmode_blocking)
    vrep.simxFinish(self.client_id)

  def reset(self):
    # Close the simulation
    self.close()
    # Restart the simulation
    self.setup_simulation()

class DynamicMovementPrimitives(object):
  """
  Class for Dynamic Movement Primitives (DMPs)
  """
  def __init__(self):
    """
    ----------------------- DO NOT MODIFY! ---------------
    Initialize hyperparameters for DMP
    """
    self.alpha = 1.0
    self.beta = 1.25
    self.tau = 1.5
    self.dt = 0.05
    self.c = np.zeros([NUM_DIM, NUM_BASIS])
    self.h = np.zeros([NUM_DIM, NUM_BASIS])

    for i in range(NUM_DIM):
      for j in range(1,NUM_BASIS+1):
        self.c[i,j-1] = np.exp(-(j - 1) * 0.5/(NUM_BASIS - 1))
      for j in range(1,NUM_BASIS):
        self.h[i,j-1] = 0.5 / (0.65 * (self.c[i,j] - self.c[i,j-1])**2)
      self.h[i,NUM_BASIS-1] = self.h[i,NUM_BASIS-2]

  def get_trajectories(self, w, g, y0, plot_traj=False):
    n_samples = w.shape[0]
    trajectories = np.zeros((n_samples, TRAJ_HORIZON, 3))
    # Each sample of w
    x = None
    for sample_id in range(n_samples):
      trajectories[sample_id, :], x = self.get_trajectory(w[sample_id, :].reshape((NUM_DIM, NUM_BASIS)), g, y0, plot_traj)

    print(bcolors.OKBLUE+'DMP solved in itertions: {}, final x: {}!'.format(TRAJ_HORIZON, x)+bcolors.ENDC)
    return trajectories

  def get_trajectory(self, w, g, y0, plot_traj = False):
    """
    ----------------------- TODO, implement DMP here ---------------

    Parameters
    ----------
    w : weigthts of the basis function, numpy array of size (NUM_DIM, NUM_BASIS)
    g : Goal position of end-effector, numpy array of size (NUM_DIM,)
    y0 : Start position of end-effector, numpy array of size (NUM_DIM,)
    plot_traj : Pass in True if you want to plot the DMP trajectory

    Returns
    ----------
    Two outputs:
    1) trajectory computed with DMPs, numpy array of size (TRAJ_HORIZON, NUM_DIM)
    2) final value of x, the timer in the canonical system (which should be close to 0 at the end of the DMP)
    """
    def forcing(x, w):
        psi = np.exp(-1 * self.h * (x - self.c) ** 2);
        return np.sum(w * psi * x, axis = 1, keepdims = 1)/(np.sum(psi, axis = 1, keepdims = 1) + 10e-8)
    
    def f(y, t):
        y = np.expand_dims(y, 1)
        xp = y[:NUM_DIM]
        x = y[NUM_DIM:2*NUM_DIM]
        v = y[-1, None]

        force = forcing(v, w)
        xpp = self.alpha * (self.beta * 1 / (self.tau ** 2) * (g[:,None] - x) - 1 / self.tau * xp) + (g[:,None] - y0[:,None]) / (self.tau ** 2) * force 
        
        ret = np.concatenate([xpp, xp, -self.tau * v], axis = 0)
        return np.squeeze(ret)

    t = np.arange(0, TRAJ_HORIZON * self.dt, self.dt)
    traj = scipy.integrate.odeint(f, np.concatenate([np.zeros_like(y0), y0, np.array([1])]), t)
    x = traj[-1, -1]
    traj = traj[:,NUM_DIM:NUM_DIM * 2] 

    if False:
      fig = plt.figure()
      ax1 = fig.add_subplot(311)
      ax2 = fig.add_subplot(312)
      ax3 = fig.add_subplot(313)

      ax1.plot(range(len(traj)), np.array(traj)[:,0], 'b')
      ax1.plot(range(len(traj)), np.repeat(g[0],len(traj)), 'r')
      ax1.set_ylabel('x')
      ax1.set_title('Trajectory')

      red_patch = mpatches.Patch(color='red', label='g')
      blue_patch = mpatches.Patch(color='blue', label='y')
      plt.legend(handles=[red_patch, blue_patch])

      ax2.plot(range(len(traj)), np.array(traj)[:,1], 'b')
      ax2.plot(range(len(traj)), np.repeat(g[1],len(traj)), 'r')
      ax2.set_ylabel('y')

      ax3.plot(range(len(traj)), np.array(traj)[:,2], 'b')
      ax3.plot(range(len(traj)), np.repeat(g[2],len(traj)), 'r')
      ax3.set_ylabel('z')
      ax3.set_xlabel('time')

      plt.show()

    return traj, x

class RelativeEntropyPolicySearch(object):
  """
  Class for Relative Entropy Policy Search (REPS)
  """
  def __init__(self, env, policy, epsilon=0.1):
    self.env = env
    self.policy = policy

    # Weights
    self.weight_mean = np.zeros(NUM_DIM * NUM_BASIS)
    self.weight_std = np.ones(NUM_DIM * NUM_BASIS)

    # REPS
    self.epsilon = epsilon
    self.returns_so_far = []

  def do_rollouts(self, num_rollouts=5):
    """
    ----------------------- TODO ---------------
    Performs rollouts based on current policy on the VREP environment

    Parameters
    ----------
    num_rollouts : The desired number of rollouts

    Returns
    ----------
    returns : a numpy array of size (num_rollouts, 1)
    w : policy parameters the rollouts were attempted for
    """
    #TODO: - Sample weights from the gaussian distribution;
    # replace the following line with the correct solution
    w = np.stack([np.random.normal(self.weight_mean, self.weight_std) for i in range(num_rollouts)])

    # Get start and goal state
    start_state = self.env.RobotPosition
    goal_state = self.env.GoalPosition
    
    # Obtain trajectories from DMP policy
    trajectories = self.policy.get_trajectories(w, goal_state, start_state)

    # Get returns of the trajectories
    returns = self.env.executeTrajectories(trajectories)

    return returns, w

  def train(self, num_iter=10, num_rollouts_per_iter=10):
    """
    Trains the DMPs with REPS

    Parameters
    ----------
    num_rollouts_per_iter : The number of rollouts of the current policy per iteration
    num_iter: The total number of iterations
    """
    mean_returns = []
    for it in range(num_iter):
      returns, w = self.do_rollouts(num_rollouts=num_rollouts_per_iter)
      mean_returns.append(np.mean(returns))
      print(bcolors.OKGREEN + 'Avg returns is ' + str(np.mean(returns)) + bcolors.ENDC)
      alpha = self.reps_update(returns)
      self.weight_mean, self.weight_std = self.fit_gaussian(alpha, w)
      print(self.weight_mean)
      print(self.weight_std)

    plt.plot(range(len(mean_returns)), mean_returns)
    plt.xlabel('Iterations')
    plt.ylabel('Avg Return')
    plt.title('Learning curve')
    plt.show()
      
  def reps_update(self, returns):
    """
    ----------------------- TODO, implement REPS here ---------------
    The core function implementing the REPS objective

    Parameters
    ----------
    returns : numpy array of size (num_rollouts, 1)

    Returns
    ----------
    alpha : an output parameter for REPS policy update,
            a numpy array of size (num_rollouts, 1)
    """
    def g(n, r):
        k = r.shape[0]
        return n * np.log(1 / k * sum(np.exp(r / n))) + n * self.epsilon
    # TODO: implement REPS
    # HINT: checkout scipy.optimize.minimize
    # replace line below with the correct solution
    Rn = returns - np.max(returns)
    n = scipy.optimize.minimize(g, 1, (Rn)) 
    alpha = np.exp(Rn / n.x)
    print(bcolors.OKGREEN + 'Alpha for REPS update is ' + str(alpha)+bcolors.ENDC)
    return alpha

  def fit_gaussian(self, alpha, w):
    """
    ----------------------- TODO ---------------
    Updates the gaussian distribution parameters (mean, std) based on w and alpha

    Parameters
    ----------
    alpha : numpy array of size (num_rollouts, 1)
    w : numpy array of size (num_rollouts, NUM_DIM * NUM_BASIS)

    Returns
    ----------
    weight_mean : numpy array of size (NUM_DIM * NUM_BASIS, 1)
    weight_std : numpy array of size (NUM_DIM * NUM_BASIS, 1)
    """
    mu = np.sum(alpha * w, axis = 0, keepdims = True) / np.sum(alpha)
    sigma = np.sum((w - mu) ** 2 * alpha, axis = 0, keepdims = True) / np.sum(alpha)  
    return np.squeeze(mu), np.squeeze(np.sqrt(sigma))

  def test(self):
    """
    Final testing function; use the mean of the distribution as the DMP policy parameter
    """
    # Use the mean weights
    w = self.weight_mean.flatten().reshape((NUM_DIM, NUM_BASIS))

    # Get start and goal state
    start_state = self.env.RobotPosition
    goal_state = self.env.GoalPosition
    
    # Get dmp trajectory
    trajectory, x = self.policy.get_trajectory(w, goal_state, start_state)

    # Execute trajectory on environment
    return_eval = self.env.executeTrajectory(trajectory)
    print(bcolors.BOLD+'At test, the return obtained is '+str(return_eval)+bcolors.ENDC)
    return return_eval


def main(args):
  env = VRepEnvironment(args.task_id)
  signal.signal(signal.SIGINT, env.signal_handler)

  planner = RRTConnectPlanner(env.obstacles)
  plan = planner.Plan(np.array([0, 0.5]), np.array([0, -0.5]))
  pdb.set_trace()
  for p in plan:
    for i in range(10):
      env.setRobotPosition(np.array([p[0], p[1], 0.5]))

  dmp_policy = DynamicMovementPrimitives()
  reps = RelativeEntropyPolicySearch(env, dmp_policy)
  reps.train(num_iter=args.n_iter, num_rollouts_per_iter=args.n_rollouts)
  reps.test()
  env.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=0, help='Use this to set the random seed')
  parser.add_argument('--n_rollouts', type=int, default=5)
  parser.add_argument('--n_iter', type=int, default=10)
  parser.add_argument('--task_id', type=int, default=0, help='0 for handle reacher (first part), 1 for the opening door (second part)')
  args = parser.parse_args()
  np.random.seed(args.seed)
  random.seed(args.seed)
  main(args)
