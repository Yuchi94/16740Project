import numpy, operator
from RRTTree import RRTTree
import numpy as np
from collections import deque


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

    for obs in self.obstacles:
      if np.linalg.norm(point - obs) < self.r:
        return True

    return False


  def backtrack(self, tree_nodes, tree_parent, n):
    """
    Backtrack.
    """

    inv_plan = [ tree_nodes[n][1] ]

    while tree_parent[n] > 0:
      n = tree_parent[n]
      inv_plan.append(tree_nodes[n][1])

    inv_plan.reverse()
    plan = inv_plan

    return plan


  def Plan(self, s_init, s_goal, epsilon = 0.10, verbose = False):
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

    A = [
      np.array([  1,    0,    0 ]),
      np.array([ -1,    0,    0 ]),
      np.array([    0,  1,    0 ]),
      np.array([    0, -1,    0 ]),
      np.array([    0,    0,  1]),
      np.array([    0,    0, -1])
    ]

    def getLocation(start, shift):
        return start + shift * epsilon / 4.0

    # initialize search tree

    tree_nodes    = [ (np.array([0,0,0]), None) ]
    tree_parent   = [ -1 ]
    tree_children = [ set() ]

    # initialize search
    visited = set()
    q = deque()
    q.append(0)

    # search
    plan = None
    while len(q) > 0:
      n = q.popleft()
      s_cur = np.array(tree_nodes[n][0])

      # update visited status
      visited.add(tuple(s_cur))

      if verbose:
        print(len(tree_nodes), len(q))

      # check goal
      if np.linalg.norm(np.array(s_goal) - getLocation(s_init, s_cur)) < epsilon:
        #plan = self.backtrack(tree_nodes, tree_parent, n)
        inv_plan = [ tree_nodes[n][1] ]

        while tree_parent[n] > 0:
          n = tree_parent[n]
          inv_plan.append(getLocation(s_init, tree_nodes[n][1]))

        plan = inv_plan.reverse()
        break

      # append valid successors
      for a in A:
        s_next = s_cur + a

        # check visited status
        if tuple(s_next) in visited:
          continue

        # check collision
        if self.checkCollision(getLocation(s_init, s_next)):
          continue

        # append to the search tree
        tree_nodes.append((np.array(s_next), a))
        tree_children.append(set())

        n_next = len(tree_nodes) - 1
        tree_parent.append(n)
        tree_children[n].add(n_next)

        # update search status
        q.append(n_next)

    return plan
