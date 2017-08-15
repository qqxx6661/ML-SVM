import numpy as np
import matplotlib.pyplot as plt

size = 4
x = np.arange(size)

#video_file_cloud_o = (80760, 103150, 136175, 165440)
video_file_cloud_o = (4.907, 5.013, 5.134, 5.219)
prediction_cloud = (1.69, 1.957, 2.52, 2.64)
prediction_EaOT = (1.58, 1.898, 2.13, 2.27)
video_file_edge_o = (4.649, 4.782, 4.949, 5.05)


total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers', fontsize=18)
plt.ylabel('Computing Cost (lg(ms))', fontsize=18)


plt.bar(x-0.45*width, video_file_cloud_o, fc='#036564', width=0.75*width, label='Video Analysis (Cloud)')
plt.bar(x-0.45*width, prediction_cloud, fc='#033649', width=0.75*width, bottom=video_file_cloud_o, label='Prediction (Cloud)')
plt.bar(x+0.45*width, video_file_edge_o, fc='#764D39', width=0.75*width, label='Video Analysis (EaOT)')
plt.bar(x+0.45*width, prediction_cloud, fc='#250807', width=0.75*width, bottom=video_file_edge_o, label='Prediction (EaOT)')

plt.xticks(x, (2, 4, 6, 8), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='center', bbox_to_anchor=(0.77, 0.13), fontsize=11)
plt.show()
