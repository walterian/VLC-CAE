import numpy as np
for delta in np.arange(1, 101):
    x = np.arange(-1,1,0.01)

    y = 1/(1 + 2.718281 ** (-x*delta))
    print(y[:])