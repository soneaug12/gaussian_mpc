import math
import numpy as np
import matplotlib.pyplot as plt

from create_track import *
from create_track2 import *

if __name__ == "__main__":

    waypoints = get_track()
    waypoints2 = get_track2()

    xs   = waypoints[:,0]
    ys   = waypoints[:,1]
    vs   = waypoints[:,4]
    dts  = waypoints[:,5]
    cs   = waypoints[:,6]
    ts   = waypoints[:,7]
    str  = waypoints[:,8]
    dstr = waypoints[:,9]
    gy   = waypoints[:,10]
    ks   = waypoints[:,11]
    phis = waypoints[:,12]

    xs2   = waypoints2[:,0]
    ys2   = waypoints2[:,1]
    vs2   = waypoints2[:,4]
    dts2  = waypoints2[:,5]
    cs2   = waypoints2[:,6]
    ts2   = waypoints2[:,7]
    str2  = waypoints2[:,8]
    dstr2 = waypoints2[:,9]
    gy2   = waypoints2[:,10]
    ks2   = waypoints2[:,11]
    phis2 = waypoints2[:,12]

    fig = plt.figure(figsize=(20,12))
    fig.add_subplot(5, 2, 1)
    plt.plot(cs,ks)
    plt.plot(cs2,ks2)
    plt.xlabel('S [m]')
    plt.ylabel('Curvature [1/m]')
    plt.grid()
    fig.add_subplot(5, 2, 3)
    plt.plot(cs,phis)
    plt.plot(cs2,phis2)
    plt.xlabel('S [m]')
    plt.ylabel('Yaw [rad]')
    plt.grid()
    fig.add_subplot(5, 2, 5)
    plt.plot(cs,xs)
    plt.plot(cs2,xs2)
    plt.xlabel('S [m]')
    plt.ylabel('X [m]')
    plt.grid()
    fig.add_subplot(5, 2, 7)
    plt.plot(cs,ys)
    plt.plot(cs2,ys2)
    plt.xlabel('S [m]')
    plt.ylabel('Y [m]')
    plt.grid()
    fig.add_subplot(5, 2, 9)
    plt.plot(ts,dstr*rad2deg)
    plt.plot(ts2,dstr2*rad2deg)
    plt.xlabel('Time [s]')
    plt.ylabel('Steering speed [deg/s]')
    plt.grid()
    fig.add_subplot(5, 2, 2)
    plt.plot(cs,vs)
    plt.plot(cs2,vs2)
    plt.xlabel('S [m]')
    plt.ylabel('V [m/s]')
    plt.grid()
    fig.add_subplot(5, 2, 4)
    plt.plot(cs,gy)
    plt.plot(cs2,gy2)
    plt.grid()
    plt.xlabel('S [m]')
    plt.ylabel('Gy [G]')
    fig.add_subplot(5, 2, 6)
    plt.plot(cs,ts)
    plt.plot(cs2,ts2)
    plt.grid()
    plt.xlabel('S [m]')
    plt.ylabel('Time [s]')
    fig.add_subplot(5, 2, 8)
    plt.plot(xs,ys)
    plt.plot(xs2,ys2)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.grid()
    fig.add_subplot(5, 2, 10)
    plt.plot(ts,str*rad2deg)
    plt.plot(ts2,str2*rad2deg)
    plt.xlabel('Time [s]')
    plt.ylabel('Steering angle [deg]')
    plt.grid()
    plt.tight_layout()
    plt.show()