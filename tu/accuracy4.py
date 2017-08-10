import numpy as np
import matplotlib.pyplot as plt

x = (1, 2, 3, 4)
y1 = (98.502, 98.640, 98.807, 99.001)
#y2 = (96.226, 97.780, 97.836, 98.335)

plt.plot(x, y1, marker='o', c='r', label='Accuracy in Scene 1')
#plt.plot(x, y2, marker='o', c='b', label='Accuracy in Scene 2')
plt.xticks(x, (900, 1800, 4500, 9000))
plt.xlabel('Total Sample Amount', fontsize=13)
plt.ylabel('Accuracy of Prediction (%)', fontsize=13)
# plt.legend(loc="lower left", bbox_to_anchor=(0.35, 0.01))
plt.legend()
plt.grid(linestyle='--')
plt.show()