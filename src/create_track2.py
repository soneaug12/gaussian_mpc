import sys
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import display, clear_output

np.set_printoptions(linewidth=200)
torch.set_printoptions(linewidth=200)

rad2deg = 180/np.pi
deg2rad = np.pi/180

def chirp_sin(t, tvec, fvec):
    f = np.interp(t,tvec, fvec)
    y = np.zeros_like(t)

    x = 0
    for i in range(len(t)):
        if i == 0:
            dt = t[i+1] - t[i]
        else:
            dt = t[i] - t[i-1]
        x += 2*np.pi*f[i]*dt
        y[i] = np.sin(x)
    
    return f, y

def get_track2(plot_on=False):
    
    cs_max = 800#1000
    ds = 0.01
    Ns = int(np.floor(cs_max/ds))+1
    cs = np.linspace(0, cs_max,num=Ns)
    ks = np.zeros_like(cs)

    # cs_vec = np.array([0,  200,  400,  600,  800,  1000])
    # fs_vec = np.array([0, 0.015, 0.03, 0.015, 0.03,  0.015])
    cs_vec = np.array([0,  200,  400,  600,  800,  1000])
    fs_vec = np.array([0, 0.014, 0.020, 0.026, 0.032,  0.016])
    f,ks = chirp_sin(cs, cs_vec, fs_vec)

    ks = ks * 0.1

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

    l = 2.7
    lf = 1.3
    lr = l - lf
    m = 2000
    Kf = 60000
    Kr = 170000
    N = 16
    cv_gain = (1-m/(2*l**2)*(lf*Kf-lr*Kr)/(Kf*Kr)*v0**2)*l*N
    str2 = ks2*cv_gain
    dstr2 = np.gradient(str2)/np.gradient(ts2)

    if plot_on:
        fig = plt.figure(figsize=(20,12))
        fig.add_subplot(5, 2, 1)
        plt.plot(cs2,ks2)
        plt.xlabel('S [m]')
        plt.ylabel('Curvature [1/m]')
        plt.grid()
        fig.add_subplot(5, 2, 3)
        plt.plot(cs2,phis2)
        plt.xlabel('S [m]')
        plt.ylabel('Yaw [rad]')
        plt.grid()
        fig.add_subplot(5, 2, 5)
        plt.plot(cs2,xs2)
        plt.xlabel('S [m]')
        plt.ylabel('X [m]')
        plt.grid()
        fig.add_subplot(5, 2, 7)
        plt.plot(cs2,ys2)
        plt.xlabel('S [m]')
        plt.ylabel('Y [m]')
        plt.grid()
        fig.add_subplot(5, 2, 9)
        plt.plot(ts2,dstr2*rad2deg)
        plt.xlabel('Time [s]')
        plt.ylabel('Steering speed [deg/s]')
        plt.grid()
        fig.add_subplot(5, 2, 2)
        plt.plot(cs2,vs2)
        plt.xlabel('S [m]')
        plt.ylabel('V [m/s]')
        plt.grid()
        fig.add_subplot(5, 2, 4)
        plt.plot(cs2,gy2)
        plt.grid()
        plt.xlabel('S [m]')
        plt.ylabel('Gy [G]')
        fig.add_subplot(5, 2, 6)
        plt.plot(cs2,ts2)
        plt.grid()
        plt.xlabel('S [m]')
        plt.ylabel('Time [s]')
        fig.add_subplot(5, 2, 8)
        plt.plot(xs2,ys2)
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.axis('equal')
        plt.grid()
        fig.add_subplot(5, 2, 10)
        plt.plot(ts2,str2*rad2deg)
        plt.xlabel('Time [s]')
        plt.ylabel('Steering angle [deg]')
        plt.grid()

        plt.tight_layout()
        plt.show()

    # waypoints = np.c_[xs2,ys2,np.gradient(xs2),np.gradient(ys2),vs2,dts2]
    waypoints = np.c_[xs2,ys2,np.gradient(xs2),np.gradient(ys2),vs2,dts2,cs2,ts2,str2,dstr2,gy2,ks2,phis2]

    return waypoints

def main():
    get_track2(True)

if __name__ == "__main__":
    main()