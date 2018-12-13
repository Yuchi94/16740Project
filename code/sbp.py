import numpy, operator
from RRTTree import RRTTree
import numpy as np


class SBPlanner():
  """
  Search-based planner.
  """

  def __init__(self, obstacles):
    """
    Initialize a planner.
    """

    self.obstacles = obstacles
    self.r = 0.1


  def checkCollision(self, point):
    """
    Check for collision.
    """

    for i in self.planning_env:
      if np.linalg.norm(point - i) < self.r:
        return True

    return False


  def Plan(self, start_config, goal_config, epsilon = 0.10):
    """
    Find a feasible plan.
    """

    # define action space
    delta = epsilon / 4.0
    A = [
      np.array([  delta,    0.0,    0.0 ]),
      np.array([ -delta,    0.0,    0.0 ]),
      np.array([    0.0,  delta,    0.0 ]),
      np.array([    0.0, -delta,    0.0 ]),
      np.array([    0.0,    0.0,  delta ]),
      np.array([    0.0,    0.0, -delta ])
    ]

    # initialize search tree
    tree_nodes    = []
    tree_parent   = []
    tree_children = []


    return [ A[t] for t in range(6) ]
