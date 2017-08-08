import numpy as np
import matplotlib.pyplot as plt

x = (1, 2, 3, 4, 5)
y1 = (38.90, 40.47, 57.89, 70, 80)
y2 = (84.72, 85.12, 89.14, 92, 95)
y3 = (24, 28, 38, 49, 60)
y4 = (73, 75, 81, 85, 90)

plt.plot(x, y1, marker='o', c='r', label='Accuracy of Classification in Scene 1')
plt.plot(x, y2, marker='o', c='r', ls='-.', label='Accuracy of Status in Scene 1')
plt.plot(x, y3, marker='o', c='b', label='Accuracy of Classification in Scene 2')
plt.plot(x, y4, marker='o', c='b', ls='-.', label='Accuracy of Status in Scene 2')
plt.xticks(x, (900, 1800, 9000, 18000, 54000))
plt.xlabel('Total Sample Amount', fontsize=13)
plt.ylabel('Accuracy of Prediction (%)', fontsize=13)
plt.legend(loc="lower left", bbox_to_anchor=(0.35, 0.01))
plt.show()