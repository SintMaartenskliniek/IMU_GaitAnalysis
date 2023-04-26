import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

def calcCalGyr(dq, Fs):
    # Compute angular velocity in sensor frame using first order approximation
    return 2*dq[:,1:]*Fs

def calcCalAcc(dv, Fs):
    # Compute acceleration in sensor frame using first order approximation
    return dv*Fs

def eulFromQuat(q_ls):
    N = len(q_ls)
    e = np.zeros((N,3))
    for i in range(0,N):
        R_ls = R.from_quat(q_ls[i,:])
        e[i,:] = R_ls.as_euler('zyx', degrees=True)
    return e

def calcFreeAcc(q_ls, acc_s):
    N = len(q_ls)
    acc_l = np.zeros((N,3))
    for i in range(0,N):
        q = Quaternion(q_ls[i,:])
        acc_l[i,:] = q.rotate(acc_s[i,:])
    acc_l[:,2] = acc_l[:,2]-9.81
    return acc_l

