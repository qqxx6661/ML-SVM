import numpy as np
import matplotlib.pyplot as plt

size = 4
x = np.arange(size)
# a = np.random.random(size)
# b = np.random.random(size)
a = (610, 1220, 1830, 2440)
b = (250, 500, 750, 1000)
# c = np.random.random(size)

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers')
plt.ylabel('Computing Cost (Sec)')
plt.bar(x, a,  width=width, label='Cloud')
plt.bar(x + width, b, width=width, label='EaOT')
# plt.bar(x + 2 * width, c, width=width, label='c')
plt.xticks(x, (2, 4, 6, 8))
plt.legend()
plt.show()
