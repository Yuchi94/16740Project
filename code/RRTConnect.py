import numpy, operator
from RRTTree import RRTTree
import numpy as np
import pdb

class RRTConnectPlanner():

    def __init__(self, planning_env):
        self.planning_env = planning_env
        self.delta = 0.005
        self.r = 0.1
        

    def Plan(self, start_config, goal_config, epsilon = 0.02):
        
        ftree = RRTTree(start_config)
        rtree = RRTTree(goal_config)
        plan = []

        goalDistThreshold = 0.15
        goalDist = np.linalg.norm(start_config - goal_config)

        sgoal = start_config.copy()
        egoal = goal_config.copy()

        while goalDist > goalDistThreshold:
            fnew_config = self.generateRandomConfig(egoal)
            fclosest_id, fclosest_config = ftree.GetNearestVertex(fnew_config)
            fnear_config = (fnew_config - fclosest_config) * min(1, (epsilon / np.linalg.norm(fclosest_config - fnew_config))) + fclosest_config
            fnew_config = self.Extend(fclosest_config, fnear_config, epsilon)

            if fnew_config is not None:

                fnew_id = ftree.AddVertex(fnew_config)
                ftree.AddEdge(fclosest_id, fnew_id)

                while True:

                    rclosest_id, rclosest_config = rtree.GetNearestVertex(fnew_config)
                    rnear_config = (fnew_config - rclosest_config) * (epsilon / np.linalg.norm(rclosest_config - fnew_config)) + rclosest_config
                    rnew_config = self.Extend(rclosest_config, rnear_config, epsilon)
                    if rnew_config is not None:
                        rnew_id = rtree.AddVertex(rnew_config)
                        rtree.AddEdge(rclosest_id, rnew_id)
                    else:
                        ftree, rtree = rtree, ftree
                        sgoal, egoal = egoal, sgoal
                        break

                    goalDist = np.linalg.norm(fnew_config - rnew_config)
                    print(goalDist)
                    if goalDist <= goalDistThreshold:
                        break

        fparent_id = fnew_id
        fparent_config = fnew_config

        fplan = []
        while fparent_id is not None:
            fplan.append(fparent_config)
            fchild_id = fparent_id
            fchild_config = fparent_config
            fparent_id, fparent_config = ftree.getParent(fchild_id)

        rparent_id = rnew_id
        rparent_config = rnew_config

        rplan = []
        while rparent_id is not None:
            rplan.append(rparent_config)
            rchild_id = rparent_id
            rchild_config = rparent_config
            rparent_id, rparent_config = rtree.getParent(rchild_id)

        if (fchild_config == start_config).all():
            plan = fplan[::-1] + rplan
        else:
            plan = rplan[::-1] + fplan

        return plan

    def generateRandomConfig(self, goal):
        if np.random.uniform(0, 1) < 0.1:
            return goal
        else:
            return [np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), np.random.uniform(0.1, 0.5)]

    def checkCollision(self, point):

        for i in self.planning_env:
            if np.linalg.norm(point - i) < self.r:
                return True

        return False

    def Extend(self, rclosest_config, rnear_config, epsilon):
        curr = rclosest_config.copy()
        while not self.checkCollision(curr) and np.linalg.norm(curr - rnear_config) < epsilon:
            curr = (rnear_config - rclosest_config) * (self.delta / np.linalg.norm(rclosest_config - rnear_config)) + curr

        return curr if np.linalg.norm(curr - rclosest_config) > self.delta else None


