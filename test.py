import sys
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import display, clear_output

import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import chart_studio.plotly as py

sys.path.append("./src")
from evals import *
from optimization import *
from gauss_update import *
from gaussian_process import *
from kinetic_model import *
from mpc import *
from create_track import *
from create_track2 import *

np.set_printoptions(linewidth=200)
torch.set_printoptions(linewidth=200)

# inaccurate process_model
p_model = vehicle_model(WB=2.5) 

# actual_car_model
real_car_model = vehicle_model(WB=1.5)

gp_model = GaussianProcess(dim_x=6,dim_y=2,var=0.01)

# define initial state
waypoints = get_track()
# waypoints = get_track2()
xs2 = waypoints[:,0]
ys2 = waypoints[:,1]
vs2 = waypoints[:,4]
dts2 = waypoints[:,5]
waypoints = torch.Tensor(waypoints)

with open('./data/new_track.pickle', 'rb') as td:
# with open('./data/track3_lat_only.pickle', 'rb') as td:
    data = pickle.load(td)
pos     = torch.Tensor(data[1][:-1])
pred    = torch.Tensor(data[0][1:,2:4])
label   = torch.Tensor(data[1][1:,2:4])
cont_dt = torch.Tensor(data[2][:-1])

x = torch.cat((pos[:,2:], cont_dt[:,:-1]), dim=1)
y = label - pred

state0 = torch.Tensor([0, 0, 0, vs2[0], 0, 0])

len_horizon = 10
TMAX = 400#200
step = 0
sigma_w = 0.01
start = 1

fig, ax = plt.subplots(figsize=(20,8))
x_ = np.arange(0,10)
y_ = np.arange(0,10)*0
_x = np.arange(0,10)
_y = np.arange(0,10)*0
l3, = ax.plot(xs2, ys2, '--', color='red')
l1, = ax.plot(_x,   _y, 'o', color='blue')
# l1 = ax.scatter(x=_x, y=_y, c=_y, cmap='Blues')
l2, = ax.plot(x_,   y_, '-o', color='green', linewidth=2)
ax.set_aspect('equal', 'box')
# ax.set_xlim([-100, 100])
# ax.set_ylim([-50, 100])

# ax.set_xlim([-30, 30])
# ax.set_xticks(np.linspace(-35, 35, 15))
# ax.set_ylim([-5, 25])
# ax.set_yticks(np.linspace(-5, 25, 7))
ax.grid()

# initialize estimate of uncertainty of control accuracy
vars0 = torch.ones(len_horizon)*0.1
_vars=vars0

# define first control input (not changing speed and steering angle) to be optimized later
_controls = torch.nn.Parameter(torch.zeros(len_horizon,2))

# define speed (constant velocity)
# TODO:waypointsからdt計算自動化
# dt = torch.ones(len_horizon,1)*0.3
dt0 = dts2[0]
dt = torch.ones(len_horizon,1)*dt0
vs = torch.LongTensor(torch.arange(len_horizon+1))

# yはこの時点ではメンバ変数へのセットのみで計算に不使用、
# forwardおよびpredict_で引用
gp_model.train(x, y)

# define gp_propagator (used to compute sequence of uncertainties of control accuracy)
gp_propagator = gaussian_propagator(p_model.forward,gp_model.forward,sigma_w)

real_path = []

state_truth = []
state_prdct = []
control_dt_truth = []

gl_time = []
gl_yofs = []

# t_array = []
# colors = []
# _t = np.array([0])

imax = int(TMAX/dt0)
for T in range(imax):
    if T>0:
        start = start_
        _controls = controls_.data.clone()

    # gl_time.append(T*0.3)
    gl_time.append(T*dt0)

    # make prediction with MPC (inaccurate_model + gaussian_process)
    mpc = GP_MPC(gp_propagator, construct_loss_3_gp, len_horizon, waypoints)
    controls_dt, path, vars_ = mpc.run(state0, _controls, vs, dt, start, _vars)
    controls = controls_dt[:,:-1]

    # GPRなしの通常MPC
    mpc2 = MPC(real_car_model,construct_loss_3,len_horizon,waypoints)
    controls_dt2,path2 = mpc2.run(state0,_controls,vs,dt,start)

    # final control signal selection
    controls_final = controls_dt[0]
    # controls_final = controls_dt2[0]

    # 暫定で前後方向無効化(横方向のみ評価)
    controls_final[0] = 0

    # measure actual state
    # state_real = real_car_model.forward(state0,controls_dt[0])
    state_real = real_car_model.forward(state0,controls_final)
    real_path.append(state_real.view(1,-1))

    state_truth.append(state_real.detach().numpy())
    control_dt_truth.append(controls_final.detach().numpy())

    path_est = p_model.forward(state0, controls_final)
    # state_prdct.append(path[0].detach().numpy())
    state_prdct.append(path_est.detach().numpy())

    # 代入するのは_controlsではなくcontrolsの誤記？
    # でも試してみると後半で発散する。何故？
    controls_ = _controls.data.clone()
    controls_[:-(step+1)] = _controls[step+1:] # 配列を前にシフト
    controls_[-(step+1):] = 0 # シフト後の配列後方は０埋め

    # 現在位置に一番近いコース上のindexを求めるstartが前回index, start_が新index、
    # 探索範囲は(start, start+L)
    # start_ = search_(state_real,waypoints,start,L=len_horizon)
    start_,yofs = search_2(state_real,waypoints,start,L=len_horizon)
    
    state0 = state_real.data.clone()
    _vars = vars_.data.clone()
    gl_yofs.append(yofs.detach().numpy())
    
    # visualization
    _x = torch.cat(real_path,dim=0)[:,0].data.numpy()
    _y = torch.cat(real_path,dim=0)[:,1].data.numpy()
    if T == 0:
        _t[0] = float(T)
    else:
        _t = np.concatenate([_t, np.array([float(T)])])
 
    x_ = path[:,0].data.numpy()
    y_ = path[:,1].data.numpy()
    
    l1.set_data(_x, _y) # real path
    # data = np.array((_x,_y)).T
    # t_dist = float(T)-_t
    # l1.set_offsets(data) # real path
    # l1.set_color(colors)
    l2.set_data(x_, y_) # MPC estimation
    clear_output(wait=True)
    # display(fig)
    plt.pause(0.01)

    s_zero = str(T).zfill(4)

    online_learning = 1
    if online_learning == 1:
        if start_ < start:
            state_truth2 = np.stack(state_truth)
            state_prdct2 = np.stack(state_prdct)
            control_dt_truth2 = np.stack(control_dt_truth)
            # data = [state_prdct2,state_truth2,control_dt_truth2]
            pos     = torch.Tensor(state_truth2[:-1])
            pred    = torch.Tensor(state_prdct2[1:,2:4])
            label   = torch.Tensor(state_truth2[1:,2:4])
            cont_dt = torch.Tensor(control_dt_truth2[:-1])
            
            x = torch.cat((pos[:,2:], cont_dt[:,:-1]), dim=1)
            y = label - pred
            gp_model.train(x, y)

    save_data = 0
    if save_data == 1:
        if T  == imax-1:
            state_truth2 = np.stack(state_truth)
            state_prdct2 = np.stack(state_prdct)
            control_dt_truth2 = np.stack(control_dt_truth)
            with open('track3_lat_only.pickle', mode='wb') as f:
                pickle.dump([state_prdct2,state_truth2,control_dt_truth2], f)
            break

gl_time = np.hstack(gl_time)
gl_ctrl = np.vstack(control_dt_truth)
gl_state = np.vstack(state_truth)
gl_yofs2 = np.hstack(gl_yofs)

# fig2, ax2 = plt.subplots(3,1,figsize=(20,8))
fig2 = plt.figure(figsize=(20,8))
fig2.add_subplot(4,2,1)
plt.plot(gl_time,gl_state[:,0])
plt.ylabel('X [m]')
plt.grid()
fig2.add_subplot(4,2,2)
plt.plot(gl_time,gl_state[:,1])
plt.ylabel('Y [m]')
plt.grid()
fig2.add_subplot(4,2,3)
plt.plot(gl_time,gl_state[:,2])
plt.ylabel('yaw angle\n[rad]')
plt.grid()
fig2.add_subplot(4,2,4)
plt.plot(gl_time,gl_state[:,3])
plt.ylabel('vehicle\nspeed [m/s]')
plt.grid()
fig2.add_subplot(4,2,5)
plt.plot(gl_time,gl_state[:,4])
plt.ylabel('steering\nangle [rad]')
plt.grid()
fig2.add_subplot(4,2,6)
plt.plot(gl_time,gl_state[:,5])
plt.ylabel('vehicle\nacceleration\n[m/s2]')
plt.grid()
fig2.add_subplot(4,2,7)
plt.plot(gl_time,gl_yofs2)
plt.ylabel('lateral\noffset\n[m]')
plt.grid()
# plt.ylim([-0.4, 0.4])
# plt.yticks([-0.4,-0.2,0,0.2,0.4])
plt.tight_layout()
plt.show()

# fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
# fig3.append_trace(
#     go.Scatter(
#         name='X',
#         x=gl_time,
#         y=gl_f2.T#gl_state[:,0]
#     ),
#     row=1, col=1
# )
# fig3.append_trace(
#     go.Scatter(
#         name='Y',
#         x=gl_time,
#         y=gl_state[:,1]
#     ),
#     row=2, col=1
# )

# # Create and add slider
# steps = []

# for i in range(0, len(fig3.data), 2):
#     step = dict(
#         method="restyle",
#         args=["visible", [False] * len(fig3.data)],
#     )
#     step["args"][1][i:i+2] = [True, True]
#     steps.append(step)

# sliders = [dict(
#     active=0,
#     currentvalue={"prefix": "Time:  "},
#     pad={"t": 50},
#     steps=steps
# )]

# fig3.update_yaxes(title_text="X [m]", range=[-100, 100],nticks=30, row=1, col=1)
# fig3.update_yaxes(title_text="Y [m]", range=[-100, 100],nticks=30, row=2, col=1)
# fig3.update_layout(sliders=sliders, title="Gaussian MPC", template ="plotly_white")
# fig3.show() 