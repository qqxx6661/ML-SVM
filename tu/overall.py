import numpy as np
import matplotlib.pyplot as plt

size = 4
x = np.arange(size)
# transmission_cloud = (12826.8, 25632, 38448, 51264)
transmission_cloud = (3.312, 3.613, 3.789, 3.914)
transmission_EaOT = (1.301, 1.663, 1.833, 1.959)
prediction_cloud = (1.69, 1.957, 2.52, 2.64)
prediction_EaOT = (1.58, 1.898, 2.13, 2.27)



total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers', fontsize=18)
plt.ylabel('Overall Cost (lg(ms))', fontsize=18)


plt.bar(x-0.45*width, transmission_cloud, fc='#036564', width=0.75*width, label='Transmission (Cloud)')
plt.bar(x-0.45*width, prediction_cloud, fc='#033649', width=0.75*width, bottom=transmission_cloud, label='Prediction (Cloud)')
plt.bar(x+0.45*width, transmission_EaOT, fc='#764D39', width=0.75*width, label='Transmission (EaOT)')
plt.bar(x+0.45*width, prediction_cloud, fc='#250807', width=0.75*width, bottom=transmission_EaOT, label='Prediction (EaOT)')

plt.xticks(x, (2, 4, 6, 8), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='center', bbox_to_anchor=(0.77, 0.13), fontsize=11)
plt.show()
