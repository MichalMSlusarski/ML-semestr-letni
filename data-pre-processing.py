import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset = pd.read_csv(r'C:/Users/mslus/ML-projects-with-Python/data-pre-processing.py')

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(1 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='volt ova time')

ax.grid()

fig.savefig("test.png")
plt.show()