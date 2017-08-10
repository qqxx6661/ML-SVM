import numpy as np
import matplotlib.pyplot as plt

size = 4
x = np.arange(size)
video_file_cloud = (56.51, 85.21, 111.59, 141.59)
video_file_cloud_o = (80.76, 103.15, 136.175, 165.44)
video_file_edge = (26.49, 45.71, 71.44, 95.76)
video_file_edge_o = (44.55, 60.52, 88.94, 112.23)


total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers', fontsize=13)
plt.ylabel('Computing Cost (Sec)', fontsize=13)
plt.bar(x-0.75*width, video_file_cloud, fc='#036564', width=0.5*width, label='Video (Cloud)')
plt.bar(x-0.25*width, video_file_cloud_o, fc='#033649', width=0.5*width, label='Video with tracking (Cloud)')
plt.bar(x+0.25*width, video_file_edge, fc='#764D39', width=0.5*width, label='Video (EaOT)')
plt.bar(x+0.75*width, video_file_edge_o, fc='#250807', width=0.5*width, label='Video with tracking (EaOT)')
plt.xticks(x, (2, 4, 6, 8))
plt.legend()
plt.show()
