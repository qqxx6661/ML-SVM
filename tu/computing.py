import numpy as np
import matplotlib.pyplot as plt

size = 4
x = np.arange(size)
video_file_cloud = (156.51, 185.21, 211.59, 241.59)
video_file_cloud_o = (180.76, 203.15, 236.175, 265.44)
video_file_edge = (126.49, 145.71, 171.44, 195.76)
video_file_edge_o = (144.55, 160.52, 188.94, 212.23)


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
