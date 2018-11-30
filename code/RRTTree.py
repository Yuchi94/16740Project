import operator
import pdb
import numpy as np

class RRTTree(object):
    
    def __init__(self, start_config):
        
        self.vertices = []
        self.vertices.append(start_config)
        self.edges = dict()

    def GetRootId(self):
        return 0

    def GetNearestVertex(self, config):
        
        dists = []
        for v in self.vertices:
            dists.append(self.computeDistance(config, v))

        vid, vdist = min(enumerate(dists), key=operator.itemgetter(1))

        return vid, self.vertices[vid]
            

    def AddVertex(self, config):
        vid = len(self.vertices)
        self.vertices.append(config)
        return vid

    def AddEdge(self, sid, eid):
        self.edges[eid] = sid

    def getParent(self, eid):

        # pdb.set_trace()

        try:
            return self.edges[eid], self.vertices[self.edges[eid]]
        except:
            return None, None

    def computeDistance(self, config, v):
        return np.linalg.norm(config - v)
