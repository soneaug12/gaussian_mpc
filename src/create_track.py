import sys
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import display, clear_output

np.set_printoptions(linewidth=200)
torch.set_printoptions(linewidth=200)

def get_track():
    ratio = 1
    
    # cs_max = 161.677
    cs_max = 161.637
    # cs_max = 161.7245*ratio
    ds = 0.01
    Ns = int(np.floor(cs_max/ds))+1
    cs = np.linspace(0, cs_max,num=Ns)
    ks = np.zeros_like(cs)
    c0 = (20-2.75)*ratio
    c1 = c0+10*ratio
    c2 = c1+30*ratio
    c3 = c2+20*ratio
    c4 = c3+7.1681469*ratio
    c5 = c4+20*ratio
    c6 = c5+30*ratio
    c7 = c6+10*ratio
    k0 = 0.1/ratio
    for i in range(len(cs)):
        if cs[i] <= c0:
            ks[i] = 0
        elif cs[i] <= c1:
            ks[i] = k0*(cs[i]-c0)/(c1-c0)
        elif cs[i] <= c2:
            ks[i] = k0
        elif cs[i] <= c3:
            ks[i] = -k0*(cs[i]-(c2+c3)/2)/((c3-c2)/2)
        elif cs[i] <= c4:
            ks[i] = -k0
        elif cs[i] <= c5:
            ks[i] = k0*(cs[i]-(c4+c5)/2)/((c5-c4)/2)
        elif cs[i] <= c6:
            ks[i] = k0
        elif cs[i] <= c7:
            ks[i] = -k0*(cs[i]-c7)/(c7-c6)
    
    phis = np.r_[0, np.cumsum(ks[:-1]*ds)]
    xs = np.r_[0, np.cumsum(ds*np.cos(phis[:-1]))]
    ys = np.r_[0, np.cumsum(ds*np.sin(phis[:-1]))]
    
    v0 = 18/3.6
    vs = np.ones_like(cs)*v0
    gy = vs**2*ks/9.80665
    
    dts = ds/vs
    ts = np.r_[0, np.cumsum(dts[:-1])]
    
    ds2 = 0.5*3
    step = int(ds2/ds)
    cs2 = cs[0::step]
    ks2 = ks[0::step]
    phis2 = phis[0::step]
    xs2 = xs[0::step]
    ys2 = ys[0::step]
    ts2 = ts[0::step]
    vs2 = vs[0::step]
    gy2 = gy[0::step]
    dts2 = ds2/vs2
        
    plot_on = 0
    if plot_on:
        fig = plt.figure(figsize=(20,12))
        fig.add_subplot(4, 2, 1)
        plt.plot(cs2,ks2)
        plt.xlabel('S [m]')
        plt.ylabel('Curvature [1/m]')
        plt.grid()
        fig.add_subplot(4, 2, 3)
        plt.plot(cs2,phis2)
        plt.xlabel('S [m]')
        plt.ylabel('Yaw [rad]')
        plt.grid()
        fig.add_subplot(4, 2, 5)
        plt.plot(cs2,xs2)
        plt.xlabel('S [m]')
        plt.ylabel('X [m]')
        plt.grid()
        fig.add_subplot(4, 2, 7)
        plt.plot(cs2,ys2)
        plt.xlabel('S [m]')
        plt.ylabel('Y [m]')
        plt.grid()
        fig.add_subplot(4, 2, 2)
        plt.plot(cs2,vs2)
        plt.xlabel('S [m]')
        plt.ylabel('V [m/s]')
        plt.grid()
        fig.add_subplot(4, 2, 4)
        plt.plot(cs2,gy2)
        plt.grid()
        plt.xlabel('S [m]')
        plt.ylabel('Gy [G]')
        fig.add_subplot(4, 2, 6)
        plt.plot(cs2,ts2)
        plt.grid()
        plt.xlabel('S [m]')
        plt.ylabel('Time [s]')
        fig.add_subplot(4, 2, 8)
        plt.plot(xs2,ys2)
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.axis('equal')
        plt.grid()
        plt.tight_layout()
        plt.show()

    waypoints = np.c_[xs2,ys2,np.gradient(xs2),np.gradient(ys2),vs2,dts2]

    return waypoints
