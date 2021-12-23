import matplotlib.pylab as plt
import numpy as np
import scipy.signal as signal
from scipy.signal import chirp, spectrogram

f1 = 0.1       # start frequency
f2 = 1         # end frequency
t2 = 100
fs = 1000      # sampling frequency
T = 100        # time duration of the sweep
# fade = [48000, 480]   # samlpes to fade-in and fade-out the input signal
# fade = [1000, 5000]   # samlpes to fade-in and fade-out the input signal
# t_fade = [10, 10]   # samlpes to fade-in and fade-out the input signal
t_fade = [0, 0]   # samlpes to fade-in and fade-out the input signal


'''generation of the signal'''
L = t2/np.log(f2/f1)                    # parametre of exp.swept-sine
t = np.arange(0,np.round(fs*T-1)/fs,1/fs)  # time axis
s = np.sin(2*np.pi*f1*L*np.exp(t/L))       # generated swept-sine signal

fade = [t_fade[0]*fs, t_fade[1]*fs]
# fade-in the input signal
if fade[0]>0:
    s[0:fade[0]] = s[0:fade[0]] * ((-np.cos(np.arange(fade[0])/fade[0]*np.pi)+1) / 2)
    # hoge1 = ((-np.cos(np.arange(fade[0])/fade[0]*np.pi)+1) / 2)

# fade-out the input signal
if fade[1]>0:
    s[-fade[1]:] = s[-fade[1]:] *  ((np.cos(np.arange(fade[1])/fade[1]*np.pi)+1) / 2)
    # hoge2 = ((np.cos(np.arange(fade[1])/fade[1]*np.pi)+1) / 2)

t = np.linspace(0, 100, int(100/0.001)+1)
w = chirp(t, f0=0.1, t1=t[-1], f1=1, method='linear', phi=-90)
plt.plot(t, w)

# plt.plot(s[0:96000])
# plt.plot(hoge2)
# plt.plot(t,s)
plt.grid()
plt.show()