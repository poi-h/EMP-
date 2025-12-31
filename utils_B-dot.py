import numpy as np
def process_signal(t, signal, gain = 1, A = 1):
    dt = np.diff(t)
    B = np.zeros_like(signal)
    for i in range(1,len(signal)):
        B[i] = B[i-1] + (signal[i]/gain)/A * dt[i-1]
    return B