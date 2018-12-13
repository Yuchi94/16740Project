import numpy, operator, pdb
from RRTTree import RRTTree
import numpy as np
from collections import deque
from itertools import count
from heapq import *

class AStar():
  """
  Search-based planner.
  """

  def __init__(self, obstacles):
    """
    Initialize a planner.
    """

    self.obstacles = obstacles
    self.r = 0.15


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
    self.nodes = dict()

    A = [
      np.array([  1,    0,    0 ]),
      np.array([ -1,    0,    0 ]),
      np.array([    0,  1,    0 ]),
      np.array([    0, -1,    0 ]),
      np.array([    0,    0,  1]),
      np.array([    0,    0, -1])
    ]

    def getSuccessors(coord):
      return [coord + x for x in A]

    def getLocation(start, shift):
      return start + shift * epsilon / 4.0

    def getHeuristic(coord, goal):
      return np.linalg.norm(np.array(goal) - getLocation(s_init, coord))

    def computeDistance(a, b):
      return np.linalg.norm(a - b) * epsilon / 4.0




    h = [] #Use a heap
    parents = {}
    parents[(0,0,0)] = (None, 0)
    curr_cost = {}
    tb = count()
    neighbors = getSuccessors(np.array([0,0,0]))
    for n in neighbors:
      heappush(h, (1 + getHeuristic(n, s_goal),
                    1, next(tb), n)) #Total cost, current cost, tiebreaker num, coord
      parents[tuple(n)] = (np.array([0,0,0]), 1)

    while True:
      #pop min element from heap
      node = heappop(h)

      #Add neighbors
      neighbors = getSuccessors(node[3])

      for n in neighbors:
        #OOB
        if (getLocation(s_init, n) < -0.5).any() or (getLocation(s_init, n) >= 0.5).any():
          continue

        #Explored
        if tuple(n) in parents:
          continue
          if parents[tuple(n)][1] <= node[1] + computeDistance(node[3], n):
            continue
          else:
            for H in h:
              if (H[3] == n).all():
                h.remove(H)
                break
            heapify(h)


            #Collision
        if self.checkCollision(getLocation(s_init, n)):
          continue

            #Reached the end
        if (np.linalg.norm(np.array(s_goal) - getLocation(s_init, n)) < epsilon):
          parents[tuple(n)] = (node[3], node[1] + computeDistance(node[3], n))
          return self.createPath(s_init, s_goal, parents, n, getLocation)

        #Add parents
        heappush(h, (node[1] + computeDistance(node[3], n)
            + getHeuristic(n, s_goal),
            node[1] + computeDistance(node[3], n), next(tb), n))
        parents[tuple(n)] = (node[3], node[1] + computeDistance(node[3], n))

    print("Should never reach here")

  def createPath(self, start, goal, parents, coord, getLocation):
    print("number of explored nodes: " + str(len(parents)))

    path = []
    path.append(goal)

    while True:
      path.append(getLocation(start, coord))
      #path.append(self.planning_env.discrete_env.GridCoordToConfiguration(coord))
      coord = parents[tuple(coord)][0]
      if coord is None:
        break

    path.append(start)
    return path[::-1]
