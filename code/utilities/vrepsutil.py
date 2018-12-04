import os, sys
sys.path.append(os.getcwd())

try:
    from lib import vrep
except:
    print ('"vrep.py" could not be imported. Check for the library file')
    import time

import numpy as np
import copy as cp
import time

############## Initialization ################

def ConnectVREPS():
    vrep.simxFinish(-1) # just in case, close all opened connections
    clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
    if clientID==-1:
        print ('Failed connecting to remote API server')
        exit()
    else:
        vrep.simxSynchronous(clientID,True)
        #print ('Connected to remote API server')
    return clientID

def GetRobotHandles(clientID):
    JointNames=["Prismatic_joint_X","Prismatic_joint_Y","Prismatic_joint_Z","Revolute_joint_Z","Revolute_joint_Y","Revolute_joint_X","Prismatic_joint_Fing1","Prismatic_joint_Fing2"]
    JointHandles=[vrep.simxGetObjectHandle(clientID,Name,vrep.simx_opmode_blocking)[1] for Name in JointNames]
    
    EENames=["EndEffector"]
    EEHandles=[vrep.simxGetObjectHandle(clientID,Name,vrep.simx_opmode_blocking)[1] for Name in EENames]
  
    FTNames=["FT_sensor_Wrist","FT_sensor_EE"]
    FTHandles=[vrep.simxGetObjectHandle(clientID,Name,vrep.simx_opmode_blocking)[1] for Name in FTNames]

    CamRGBNames=["PalmCam_RGB"]
    CamRGBHandles=[vrep.simxGetObjectHandle(clientID,Name,vrep.simx_opmode_blocking)[1] for Name in CamRGBNames]
  
    CamDepthNames=["PalmCam_Depth"]  
    CamDepthHandles=[vrep.simxGetObjectHandle(clientID,Name,vrep.simx_opmode_blocking)[1] for Name in CamDepthNames]    
  
    RobotHandles={"Joint":JointHandles,"EE":EEHandles,"FT":FTHandles,"RGB":CamRGBHandles,"Depth":CamDepthHandles};
  
  
    ## Start Data Streaming
    JointPosition=[vrep.simxGetJointPosition(clientID, jointHandle, vrep.simx_opmode_streaming)[1] for jointHandle in RobotHandles["Joint"]];
    EEPose=vrep.simxGetObjectPosition(clientID,RobotHandles["EE"][0],-1,vrep.simx_opmode_streaming)[1];
    EEPose.extend(vrep.simxGetObjectOrientation(clientID,RobotHandles["EE"][0],-1,vrep.simx_opmode_streaming)[1]);      
    FTVal=[vrep.simxReadForceSensor(clientID,FTHandle,vrep.simx_opmode_streaming)[2:4] for FTHandle in RobotHandles["FT"]];
    
    RobotVisionRGB=[vrep.simxGetVisionSensorImage(clientID,sensorHandle,0,vrep.simx_opmode_streaming) for sensorHandle in RobotHandles["RGB"]];
    RobotVisionDepth=[vrep.simxGetVisionSensorDepthBuffer(clientID,sensorHandle,vrep.simx_opmode_streaming) for sensorHandle in RobotHandles["Depth"]];
    
    return RobotHandles


############## Generate Scene and Get Object Handles ################

def GenBasicObjDesc():
    ObjDesc=[]
    
    DynObj=[]
    DynObj.extend([1.0,
                   0.05,0.05,0.05,
                   0.0,0.0,0.2,
                   0.,0.,0.,
                   0.2])
    ObjDesc.append(DynObj)
    
    DynObj=[]
    DynObj.extend([1.0,
                   0.05,0.05,0.05,
                   0.2,0.0,0.2,
                   0.,0.,0.,
                   0.2])
    DynObj.extend([1.0,
                   0.01,0.01,0.1,
                   0.3,0.0,0.2,
                   0.,0.,0.,
                   0.2])
    ObjDesc.append(DynObj)
    
    return ObjDesc, []

def GenConGridObjDesc(seed=0):
    ObjDesc=[]
    
    # IF you want to control the environment produced
    rng = np.random.RandomState(seed=seed)
    
    Blocks = np.zeros((4, 4))
    block_id = 0
    for i in range(4):
        for j in range(4):
            Blocks[i, j] = block_id
            block_id += 1

    Blocks_old = Blocks.copy()
    num_samples = 20
    count = 0
    while count < num_samples:
        pos = rng.randint(low=0, high=4, size=2)
        x, y = pos
        # first toss
        toss = rng.random_sample()
        if True:
            # Do nothing
            count += 1
            continue
        # Else connect neighbors
        # but first check if two neighbors around it already have the same value
        block_val = Blocks[x, y]
        count_same = 0
        if x!=3:
            count_same += int(block_val == Blocks[x+1, y])
        if x!=0:
            count_same += int(block_val == Blocks[x-1, y])
        if y!=3:
            count_same += int(block_val == Blocks[x, y+1])
        if y!=0:
            count_same += int(block_val == Blocks[x, y-1])

        if count_same >= 2:
            # Do nothing
            count += 1
            continue
        # Else then we connect any random neighbor with the current block
        while True:
            idx = rng.randint(low=0, high=2, size=2)
            if not (idx[0] == 0 or idx[1] == 0):
                continue
            if x + idx[0] < 0 or x + idx[0] > 3:
                continue
            if y + idx[1] < 0 or y + idx[1] > 3:
                continue
            # Valid neighbor
            Blocks[x + idx[0], y + idx[1]] = block_val
            break

    # print(Blocks)
    # print
    # print('          ^    ')
    # print('          |    ')
    # print
    # print(Blocks_old)
    vrep_blocks = []
    obstacles = []
    NewBlocks = np.zeros((4, 4))
    unique_block_ids = np.unique(Blocks.flatten())
    # def construct_block(i, j):

    #     # the first number controls what they obstacles are???
    #     blk = [3, 0.055,0.055,0.695, -0.3+float(i)*.2 + np.random.uniform(0, 0.1),-0.3+float(j)*.2 + np.random.uniform(0, 0.1),0.35, 0.,0.,0.,0.2]
    #     obstacles.append(np.array(blk[4:6]))
    #     return blk

    # c = 1
    # np.random.seed(0)
    # for block_id in unique_block_ids:
    #     xs, ys = np.where(Blocks == block_id)
    #     num = xs.shape[0]
    #     a = []
    #     for i in range(num):
    #         NewBlocks[xs[i], ys[i]] = c
    #         c += 1
    #         a = a + construct_block(xs[i], ys[i])
    #     vrep_blocks.append(a)
            
    # num_blocks = len(vrep_blocks)

    # DynObj=[]
    # for i in range(num_blocks):
    #     ObjDesc.append(vrep_blocks[i])
      
    return ObjDesc, [], obstacles


  
def GenBasicDoorObjDesc():
    ObjDesc=[]

    # The first index is the shape type. If it is 1, then it is a cuboid
    # The second, third and fourth index represent the dimensions
    # The fifth, sixth and seventh index represent the positions
    # The eighth, ninth and tenth index represent the orientation
    # The eleventh index represents mass
    DynObj=[]
    DynObj.extend([1.0,
                   0.25,0.005,0.5,
                   0.0,-0.5+0.0025,0.3,
                   0.,0.,0.,
                   0.1])    
    DynObj.extend([1.0,
                   0.05,0.005,0.05,
                   # 0.0,-0.5+0.05025,0.26,
                   0.0,-0.5+0.10025,0.26,
                   0.,0.,0.,
                   0.1])
    ObjDesc.append(DynObj)
    
    JointDesc=[]
    JointDesc.append([0, 0, 0,   0.125,-0.5+0.0025,0.26,     0*1.57,0,0.])
  
    return ObjDesc,JointDesc  

def GenObjects(clientID,ObjDesc):
    ObjectHandles=[]
    BlockHandles=[]
    AllHandles=[]
    for i in range(len(ObjDesc)):
        res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(clientID,'WorldCreator',vrep.sim_scripttype_childscript,'createDynObject',[],ObjDesc[i],[],bytearray(),vrep.simx_opmode_blocking) 
        ObjectHandles.extend([retInts[0]])
        if len(retInts[1:])==1:
            BlockHandles.extend(retInts[1:])    
        else:
            BlockHandles.extend(retInts[1:])
            AllHandles.append(retInts)
            
    ObjHandles={"Object":ObjectHandles, "Block":BlockHandles, "All":AllHandles}

    for i in range(len(ObjHandles["Object"])):
        ObjectPose=vrep.simxGetObjectPosition(clientID,ObjHandles["Object"][i],-1,vrep.simx_opmode_streaming)[1]
        ObjectPose.extend(vrep.simxGetObjectOrientation(clientID,ObjHandles["Object"][i],-1,vrep.simx_opmode_streaming)[1])
      
    for i in range(len(ObjHandles["Block"])):
        BlockPose=vrep.simxGetObjectPosition(clientID,ObjHandles["Block"][i],-1,vrep.simx_opmode_streaming)[1]
        BlockPose.extend(vrep.simxGetObjectOrientation(clientID,ObjHandles["Block"][i],-1,vrep.simx_opmode_streaming)[1])
      
    return ObjHandles

def GenJoints(clientID,ObjHandles,JointDesc):
    JointHandles=[]  
    for i in range(len(JointDesc)):
        intData=[int(JointDesc[i][0]),   ObjHandles["Object"][int(JointDesc[i][1])], ObjHandles["Object"][int(JointDesc[i][2])] ];
        floatData=JointDesc[i][3:];
        res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(clientID,'WorldCreator',vrep.sim_scripttype_childscript,'createDynJoint',intData,floatData,[],bytearray(),vrep.simx_opmode_blocking)
        JointHandles.extend(retInts)    
      
    return JointHandles



############## Get State Information ################

def GetRobotState(clientID,RobotHandles):
    JointPosition=[vrep.simxGetJointPosition(clientID, jointHandle, vrep.simx_opmode_buffer)[1] for jointHandle in RobotHandles["Joint"]];

    EEPose=vrep.simxGetObjectPosition(clientID,RobotHandles["EE"][0],-1,vrep.simx_opmode_buffer)[1];
    EEPose.extend(vrep.simxGetObjectOrientation(clientID,RobotHandles["EE"][0],-1,vrep.simx_opmode_buffer)[1]);
    
    FTVal=[vrep.simxReadForceSensor(clientID,FTHandle,vrep.simx_opmode_buffer)[2:4] for FTHandle in RobotHandles["FT"]];
    
    RobotState={"JointPos":JointPosition,"EEPose":EEPose,"FTVal":FTVal}
    return RobotState



def GetObjState(clientID,ObjHandles):
    ObjectPoses=[]
    for i in range(len(ObjHandles["Object"])):
        ObjectPose=vrep.simxGetObjectPosition(clientID,ObjHandles["Object"][i],-1,vrep.simx_opmode_buffer)[1]
        ObjectPose.extend(vrep.simxGetObjectOrientation(clientID,ObjHandles["Object"][i],-1,vrep.simx_opmode_buffer)[1])
        ObjectPoses.append(ObjectPose)
      
    BlockPoses=[]
    for i in range(len(ObjHandles["Block"])):
        BlockPose=vrep.simxGetObjectPosition(clientID,ObjHandles["Block"][i],-1,vrep.simx_opmode_buffer)[1]
        BlockPose.extend(vrep.simxGetObjectOrientation(clientID,ObjHandles["Block"][i],-1,vrep.simx_opmode_buffer)[1])
        BlockPoses.append(BlockPose)
      
    ObjState={"Object":ObjectPoses,"Block":BlockPoses}
    return ObjState  



def GetRobotVision(clientID,RobotHandles):
    RobotVisionRGBRaw=[vrep.simxGetVisionSensorImage(clientID,sensorHandle,0,vrep.simx_opmode_buffer)[2] for sensorHandle in RobotHandles["RGB"]]    
    RobotVisionDepthRaw=[vrep.simxGetVisionSensorDepthBuffer(clientID,sensorHandle,vrep.simx_opmode_buffer)[2] for sensorHandle in RobotHandles["Depth"]]

    RobotVisionRGB=[]
    for i in range(len(RobotVisionRGBRaw)):
        img=np.array(RobotVisionRGBRaw,dtype=np.uint8)
        img.resize([480,640,3])
        RobotVisionRGB.append(img)
      
    RobotVisionDepth=[]
    for i in range(len(RobotVisionDepthRaw)):
        img=np.array(RobotVisionDepthRaw)*.6
        img.resize([480,640])
        RobotVisionDepth.append(img) 

    RobotVision={"RGB":RobotVisionRGB,"Depth":RobotVisionDepth}
    return RobotVision


############## Control Loop ################


def SetDesJointPos(clientID,RobotHandles,DesJoints):
    vrep.simxPauseCommunication(clientID,1);
    for j in range(len(DesJoints)):
        vrep.simxSetJointTargetPosition(clientID,RobotHandles["Joint"][j],DesJoints[j],vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(clientID,0);
    return




def StepSim(clientID, nrSteps=1):
    for i in range(nrSteps):
        vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
    return





