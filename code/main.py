#!/usr/bin/env python
"""
16740-A: HW2
Skill Learning
"""
import argparse, math, os, random, sys, signal, time, scipy.integrate, scipy.optimize, pdb
import time
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
from sbp import SBPlanner
from astar import AStar
import pretrain_nnpolicy

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
    # self.object_handles = vu.GenObjects(self.client_id, object_desc, 0)

    object_desc, joint_desc = vu.GenBasicDoorObjDesc()
    self.object_handles = vu.GenObjects(self.client_id, object_desc)

    self.joint_handles = vu.GenJoints(self.client_id, self.object_handles, joint_desc)
    self.robot_handles = vu.GetRobotHandles(self.client_id)

    self.door_id = 0
    self.doorhandle_id = 1

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


def run_collect_dataset(planner, s_init_range_scale, s_init_range_min, s_goal, epsilon, N):
  """
  Collect dataset.
  """

  import pickle
  from tqdm import tqdm
  import datetime

  dataset = []

  print('collecting dataset...')
  #for i in tqdm(range(N)):
  for i in range(N):
    print('collecting sample trajectory @ %d/%d' % (i+1, N))

    plan = None
    s_init = None
    while plan is None:
      # define random initial state
      s_init = np.random.random(3) * s_init_range_scale + s_init_range_min

      # plan for goal
      try:
        plan = planner.Plan(s_init, s_goal)
      except Exception as e:
        plan = None

    # add to the dataset
    s_cur = s_init
    for t, a in enumerate(plan):
      dataset.append((s_cur, a))
      s_cur = s_cur + a

  # save the dataset
  timestamp = datetime.datetime.now().timestamp()
  save_file_name = 'dataset.' + str(timestamp) + '.pkl'
  with open(save_file_name, 'wb') as f:
    pickle.dump(dataset, f)
  print('dataset saved to %s' % (save_file_name))

  return dataset


def run_test_planner(env, planner, s_init_range_scale, s_init_range_min, s_goal, epsilon):
  """
  Test planner.
  """

  # define starting state
  s_init = np.random.random(3) * s_init_range_scale + s_init_range_min

  # set the end-effector to the initial state
  s_cur = s_init
  for i in range(10):
    print('-> %s' % (str(s_cur)))
    env.setRobotPosition(s_cur.copy())

  # find a plan
  print('planning...')
  print('s_init  = %s' % (str(s_init)))
  print('s_goal  = %s' % (str(s_goal)))
  print('epsilon = %f' % (epsilon))
  print('step    = %f' % (epsilon / 4.0))
  plan = planner.Plan(s_init, s_goal)

  input('press any key to continue...')

  # execute the plan
  print('executing...')
  for t, a in enumerate(plan):
    print('execute %s @ %d' % (str(a), t))
    s_cur = s_cur + a
    for i in range(10):
      print('-> %s' % (str(s_cur)))
      env.setRobotPosition(s_cur.copy())

  input('press any key to continue...')


def run_test_policy(env, policy, s_init_range_scale, s_init_range_min, s_goal, epsilon):
  """
  Test planner.
  """

  scalar = 100
  T = 100

  # define starting state
  s_init = np.random.random(3) * s_init_range_scale + s_init_range_min

  # set the end-effector to the initial state
  s_cur = s_init
  for i in range(10):
    print('-> %s' % (str(s_cur)))
    env.setRobotPosition(s_cur.copy())

  # find a plan
  input('press any key to continue...')

  # execute the plan
  print('executing...')
  for t in range(T):
    U_pred, Sigma_pred = policy.predict([s_cur])
    a = U_pred[0] / scalar
    print('execute %s @ %d' % (str(a), t))
    s_cur = s_cur + a
    for i in range(10):
      print('-> %s' % (str(s_cur)))
      env.setRobotPosition(s_cur.copy())

  input('press any key to continue...')


def main(args):
  """
  Launch.
  """

  print('pretraining policy...')
  policy = pretrain_nnpolicy.pretrain(
    epochs = 300,
    savedir_policy_net = 'saved/policy_net',
    savedir_variance_net = 'saved/variance_net',
    retrain = True,
    keep_training = False,
    print_test=True
  )

  print('initializing...')

  env = VRepEnvironment(args.task_id)
  signal.signal(signal.SIGINT, env.signal_handler)
  np.random.seed(int(time.time()))

  # initialize planner
  #planner = RRTConnectPlanner(env.obstacles)
  #planner = SBPlanner(env.obstacles)
  planner = AStar(env.obstacles)

  # define parameters
  s_init_range_scale = np.array([1.0, 0.0, 0.5])
  s_init_range_min   = np.array([-0.5, 0.5, 0.0])
  s_goal             = np.array([0, -0.5, 0.25])
  epsilon            = 0.10
  N_dataset_samples  = 1000

  # test planner
  #run_test_planner(env, planner, s_init_range_scale, s_init_range_min, s_goal, epsilon)

  # test policy
  run_test_policy(env, policy, s_init_range_scale, s_init_range_min, s_goal, epsilon)

  env.close()

  # collect dataset
  #run_collect_dataset(planner, s_init_range_scale, s_init_range_min, s_goal, epsilon, N_dataset_samples)


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
