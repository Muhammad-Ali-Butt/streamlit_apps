import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(i):
    x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    y = np.sin(i*x) + np.sin(2*i*x)
    line.set_data(x, y)
    return line,


fig, ax = plt.subplots()
x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
y = np.sin(x) + np.sin(2*x)
line, = ax.plot(x, y, color='red', lw=2)
ani = FuncAnimation(fig, animate, frames=100, interval=20, blit=True)
ax.set_ylim(-3, 3)
plt.title("Heart Attack Animation")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mm)")
plt.show()
