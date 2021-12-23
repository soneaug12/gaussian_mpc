import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plti
import matplotlib.animation

s = [1,2,3,4,5,6]
x = [525,480,260,300,400,600]
y = [215,180,180,100,200,200]

img = plti.imread('/home/sone/Downloads/hoge.png')
# img = plti.imread('myimage.png')
fig, ax = plt.subplots()
plt.imshow(img)
plt.axis('off')

x_vals = []
y_vals = []
iterations = len(x)
colors = []

t_vals = np.linspace(0,iterations-1,iterations,dtype=int)
scatter = ax.scatter(x_vals, y_vals, s=100, color=colors, vmin=0, vmax=1)

def init():
    pass

def update(t):
    global x, y, x_vals, y_vals
    x_vals.extend([x[t]])
    y_vals.extend([y[t]])
    scatter.set_offsets(np.c_[x_vals,y_vals])

    if t > 0:
        if s[t-1] == 1:
            colors[t-1] = [1,0,0,0.5]
        elif s[t-1] == 2:
            colors[t-1] = [0,1,0,0.5]
        else:
            colors[t-1] = [0,0,1,0.5]

    if s[t] == 1:
        colors.extend([[1,0,0,1]])
    elif s[t] == 2:
        colors.extend([[0,1,0,1]])
    else:
        colors.extend([[0,0,1,1]])
    scatter.set_color(colors)

    return ani

ani = matplotlib.animation.FuncAnimation(fig, update, frames=t_vals, init_func=init, interval=1000, repeat=False)
plt.show()