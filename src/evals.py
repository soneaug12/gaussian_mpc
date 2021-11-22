import torch

ql=1.
qc=1.
cv = 0.5

def construct_loss_(path,refs,vs,variances):
    mean_error = error_deviation_parallel_(path,refs,ql,qc)
    var_error = ( variances*(path[:,:2]-refs[:,:2]).pow(2).sum(dim=1) ).sum()
    return mean_error + cv*var_error

def construct_loss_2(path,refs,vs):
    mean_error = error_deviation_parallel_(path,refs,ql,qc)
    return mean_error

def construct_loss_3_gp(path, refs, vs, variances):
    mean_error = error_deviation_parallel_2(path,refs,ql,qc)
    var_error = ( variances*(path[:,:2]-refs[:,:2]).pow(2).sum(dim=1) ).sum()
    return mean_error + cv*var_error

def construct_loss_3(path,refs,vs):
    # x,y,v = path[:,0], path[:,1], path[:,3]
    # cx,cy,gx,gy,cv = refs[:,0],refs[:,1],refs[:,2],refs[:,3],refs[:,4]
    # ec =  gy*(x-cx) - gx*(y-cy)
    # delta_v = v - cv
    # err = qc*ec.pow(2).sum() + ql*delta_v.pow(2).sum()
    mean_error = error_deviation_parallel_2(path,refs,ql,qc)
    return mean_error

def error_deviation_parallel_(states,refs,ql,qc):
    x,y = states[:,0],states[:,1]
    cx,cy,gx,gy = refs[:,0],refs[:,1],refs[:,2],refs[:,3]
    el = -gx*(x-cx) - gy*(y-cy)
    ec =  gy*(x-cx) - gx*(y-cy)
    err = ql*el.pow(2).sum()+qc*ec.pow(2).sum()
    return err

def error_deviation_parallel_2(states,refs,ql,qc):
    x,y,v = states[:,0], states[:,1], states[:,3]
    cx,cy,gx,gy,cv = refs[:,0],refs[:,1],refs[:,2],refs[:,3],refs[:,4]
    ec =  gy*(x-cx) - gx*(y-cy)
    delta_v = v - cv
    err = qc*ec.pow(2).sum() + ql*delta_v.pow(2).sum()
    return err

def search_(state,waypoints,_indx,L):
    x,y = state[0],state[1]
    cx,cy= waypoints[:,0],waypoints[:,1]
    dx = x-cx[_indx:_indx+L]
    dy = y-cy[_indx:_indx+L]
    d2 = dx.pow(2) + dy.pow(2)
    indx = torch.argmin(d2) + _indx
    return indx


def search_2(state,waypoints,_idx,L):
    x,y = state[0],state[1]
    cx,cy= waypoints[:,0],waypoints[:,1]
    if (_idx+L) <= len(cx)-1:
        dx = x-cx[_idx:(_idx+L)]
        dy = y-cy[_idx:(_idx+L)]
        d2 = dx.pow(2) + dy.pow(2)
        idx = _idx + torch.argmin(d2)
        idx2 = idx
    else:
        cx2 = torch.cat([cx, cx])
        cy2 = torch.cat([cy, cy])
        dx = x-cx2[_idx:(_idx+L)]
        dy = y-cy2[_idx:(_idx+L)]
        d2 = dx.pow(2) + dy.pow(2)
        idx = _idx + torch.argmin(d2)
        if idx >= len(cx):
            idx2 = idx - len(cx)
        else:
            idx2 = idx

    if idx2 >= len(cx)-1:
        idx3 = idx2 - 1
    else:
        idx3 = idx2

    a =   cy[idx3+1] - cy[idx3]
    b = - cx[idx3+1] + cx[idx3]
    c =   cx[idx3+1] * cy[idx3] - cx[idx3] * cy[idx3+1]
    yofs =  (a*x + b*y + c) / torch.sqrt(a**2 + b**2)

    return idx2, yofs
