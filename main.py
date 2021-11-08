# %%

import sys
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import display, clear_output

sys.path.append("./src")
from evals import *
from optimization import *
from gauss_update import *
from gaussian_process import *
from kinetic_model import *
from mpc import *

np.set_printoptions(linewidth=200)
torch.set_printoptions(linewidth=200)

# x, y, dx, dyの順番
with open('./data/waypoints.binaryfile', 'rb') as td:
    waypoints = pickle.load(td)

cx,cy = waypoints[:,0],waypoints[:,1]

with open('./data/training_data.binaryfile', 'rb') as td:
    data = pickle.load(td)
pos = torch.cat([d[0].view(1,-1) for d in data ] , dim=0)
label = torch.cat([d[1].view(1,-1) for d in data ] , dim=0)
pred = torch.cat([d[2].view(1,-1) for d in data ] , dim=0)
cont_dt = torch.cat([d[3].view(1,-1) for d in data ] , dim=0) # control(delta, a)とdtが結合
x = torch.cat([pos[:,2:],cont_dt[:,:-1]],dim=1) #yaw, v, delta, a, d_delta, d_aの6変数
label_ = label[:,2:4]
pred_ = pred[:,2:4]
y = (label_ - pred_) # yはyawとvに関する誤差モデルで定義

gp_model = GaussianProcess(dim_x=6,dim_y=2,var=0.01)

# yはこの時点ではメンバ変数へのセットのみで計算に不使用、
# forwardおよびpredict_で引用
gp_model.train(x,y)

# inaccurate process_model
p_model = vehicle_model(WB=2.5) 

# actual_car_model
real_car_model = vehicle_model(WB=1.5)

len_horizon = 10
TMAX = 200
step = 0
sigma_w = 0.01
start = 1
speed = 3

fig, ax = plt.subplots()
x_ = np.arange(0,10)
y_ = np.arange(0,10)
_x = np.arange(0,10)
_y = np.arange(0,10)
l1, = ax.plot(_x, _y, 'o', color='blue')
l2, = ax.plot(x_, y_, 'o', color='green')
l3, = ax.plot(cx, cy,      color='red')

# define gp_propagator (used to compute sequence of uncertainties of control accuracy)
gp_propagator = gaussian_propagator(p_model.forward,gp_model.forward,sigma_w)

# define initial state
x0,y0,gx0,gy0 = waypoints[start]
# gx, gyはvx, vyの事で合っている？
# speedは定数倍しているだけ？
v0 = torch.sqrt(gx0.pow(2)+gy0.pow(2))*speed
yaw0 = torch.atan2(gy0,gx0)
delta0=torch.zeros(1)
a0=torch.zeros(1)
state0 = torch.cat( [x0.view(1),y0.view(1),yaw0.view(1),v0.view(1),delta0,a0])

# initialize estimate of uncertainty of control accuracy
vars0 = torch.ones(len_horizon)*0.1
_vars=vars0

# define first control input (not changing speed and steering angle) to be optimized later
_controls = torch.nn.Parameter(torch.zeros(len_horizon,2))

# define speed (constant velocity) 
dt = torch.ones(len_horizon,1)
vs = torch.LongTensor(torch.arange(len_horizon+1))*speed

real_path = []

for T in range(TMAX):
    if T>0:
        start = start_
        _controls = controls_.data.clone()
        
    # make prediction with MPC (inaccurate_model + gaussian_process)
    mpc = GP_MPC(gp_propagator,construct_loss_,len_horizon,waypoints)
    # mpc = MPC(p_model,construct_loss_2,len_horizon,waypoints)

    controls_dt,path,vars_ = mpc.run(state0,_controls,vs,dt,start,_vars)
    # controls_dt,path = mpc.run(state0,_controls,vs,dt,start)
    controls = controls_dt[:,:-1]
    
    # measure actual state 
    state_real = real_car_model.forward(state0,controls_dt[0])
    real_path.append(state_real.view(1,-1))

    # 代入するのは_controlsではなくcontrolsの誤記？
    # でも試してみると後半で発散する。何故？
    controls_ = _controls.data.clone()
    controls_[:-(step+1)] = _controls[step+1:] # 配列を前にシフト
    controls_[-(step+1):] = 0 # シフト後の配列後方は０埋め

    # 現在位置に一番近いコース上のindexを求めるstartが前回index, start_が新index、
    # 探索範囲は(start, start+L)
    start_ = search_(state_real,waypoints,start,L=(speed+1)*len_horizon)
    
    state0 = state_real.data.clone()
    _vars = vars_.data.clone()
    
    # visualization
    _x = torch.cat(real_path,dim=0)[:,0].data.numpy()
    _y = torch.cat(real_path,dim=0)[:,1].data.numpy()
    
    x_ = path[:,0].data.numpy()
    y_ = path[:,1].data.numpy()
    
    l1.set_data(_x[::1], _y[::1])
    l2.set_data(x_[::1], y_[::1])
    clear_output(wait=True)
    display(fig)

    s_zero = str(T).zfill(4)

    plt.pause(0.01)