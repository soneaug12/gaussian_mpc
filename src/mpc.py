import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

learning_rate=0.01
dim_s=2
EMAX=100#1000

class GP_MPC:
    def __init__(self,gp_propagator,evaluate,len_horizon,waypoints):
        self.waypoints = waypoints
        self.len_horizon = len_horizon
        self.gp_propagator = gp_propagator
        self.evaluate = evaluate
        
    def run(self,state_init,controls_init,vs,dt,start,_vars):
        start = torch.LongTensor([start])
        state_init = state_init.data.clone()
        controls = torch.nn.Parameter(controls_init.data.clone())
        vs = torch.LongTensor(vs.data.clone())
        if len(dt)==1:
            dt = torch.ones(self.len_horizon).view(-1,1)*dt
        opt =  torch.optim.Adam([controls],lr=learning_rate)

        # refは参照点。speed間隔刻みで、horizonステップ数分。初期値はstartでずらしていく
        # refs = self.waypoints[vs[1:]+start]
        if (start+self.len_horizon) <= len(self.waypoints[:,0])-1:
            refs = self.waypoints[vs[1:]+start]
        else:
            new_waypoints = torch.cat((self.waypoints, self.waypoints))
            refs = new_waypoints[vs[1:]+start]
        print(start)

        _,sigma_init = self.gp_propagator.initialize(state_init,controls[0])

        idx = np.arange(EMAX)
        ans = np.zeros(len(idx))
        for epoch in range(EMAX):
            controls_dt = torch.cat([controls,dt],dim=1)
            state  = state_init
            
            path = []
            if epoch == EMAX-1:
                vars_ = []
                sigma = sigma_init
                for t in range(self.len_horizon):
                    state,sigma = self.gp_propagator.forward(state,controls_dt[t],sigma,True)
                    sigma_xy = sigma[:dim_s,:dim_s]
                    _,eigs,_ = sigma.svd() # svdは特異値分解
                    var = eigs[0]                   # the largest eigenvalue of sigma_xy 
                    path.append(state.view(1,-1))
                    vars_.append(var.view(1))
                path = torch.cat(path,dim=0)
                vars_ = torch.cat(vars_,dim=0)
            else:
                # horizonステップ分だけ運動方程式を更新
                for t in range(self.len_horizon):
                    state,_ = self.gp_propagator.forward(state,controls_dt[t],None,False)
                    path.append(state.view(1,-1))
                path = torch.cat(path,dim=0)

            # MPCの最適化
            opt.zero_grad()
            loss = self.evaluate(path,refs,vs,_vars) 
            ans[epoch] = loss.data.numpy()
            loss.backward(retain_graph=True)
            opt.step()
            opt.zero_grad()

        controls_dt = torch.cat([controls,dt],dim=1)
        return controls_dt,path,vars_.data.clone()
    
    
    
    
class MPC:
    def __init__(self,model,evaluate,len_horizon,waypoints):

        self.waypoints = waypoints
        self.len_horizon = len_horizon
        self.model = model
        self.evaluate = evaluate
        
    def run(self,state_init,controls_init,vs,dt,start):
        start = torch.LongTensor([start])
        state_init = state_init.data.clone()
        controls = torch.nn.Parameter(controls_init.data.clone())
        vs = torch.LongTensor(vs.data.clone())
        if len(dt)==1:
            dt = torch.ones(self.len_horizon).view(-1,1)*dt
        opt =  torch.optim.Adam([controls],lr=learning_rate)

        # refs = self.waypoints[vs[1:]+start]
        if (start+self.len_horizon) <= len(self.waypoints[:,0])-1:
            refs = self.waypoints[vs[1:]+start]
        else:
            new_waypoints = torch.cat((self.waypoints, self.waypoints))
            refs = new_waypoints[vs[1:]+start]

        idx = np.arange(EMAX)
        ans = np.zeros(len(idx))

        # fig2, ax2 = plt.subplots(figsize=(20,8))
        # x_ = np.arange(0,self.len_horizon)
        # y_ = np.arange(0,self.len_horizon)
        # l1, = ax2.plot(x_, y_, 'o', color='green')
        # l2, = ax2.plot(self.waypoints[:,0], self.waypoints[:,1], color='k')
        # ax2.set_aspect('equal', 'box')

        for epoch in range(EMAX):
            controls_dt = torch.cat([controls,dt],dim=1)
            state  = state_init
            path = []
            for t in range(self.len_horizon):
                # state = self.model(state,controls_dt[t])
                state = self.model.forward(state,controls_dt[t])
                path.append(state.view(1,-1))
            path = torch.cat(path,dim=0)
     
            opt.zero_grad()
            
            loss = self.evaluate(path,refs,vs) 
            loss.backward()
            opt.step()
            ans[epoch] = loss.data.numpy()

            # l1.set_data(path[:,0].detach().numpy(), path[:,1].detach().numpy())
            # clear_output(wait=True)
            # display(fig2)
            # plt.pause(0.01)
        controls_dt = torch.cat([controls,dt],dim=1)
        # plt.close()
        return controls_dt,path