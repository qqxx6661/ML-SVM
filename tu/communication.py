import numpy as np
import matplotlib.pyplot as plt

size = 4
x = np.arange(size)
video_file = (14.2, 28.4, 42.6, 56.8)
# video_file = (1.152, 1.453, 1.629, 1.754)
data_to_cam = (0.04, 0.08, 0.12, 0.16)
data_to_cloud = (0.12, 0.24, 0.36, 0.48)
# data_to_edge = (18, 36, 54, 72)


total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers', fontsize=13)
plt.ylabel('Communication Cost (KB)', fontsize=13)
plt.bar(x-0.5*width, video_file,  width=width, label='Video file (Cloud)')
plt.bar(x-0.5*width, data_to_cam, width=width, bottom=video_file, label='Feedback (Cloud)')
plt.bar(x+0.5*width, data_to_cloud, width=width, label='Structured data (EaOT)')
plt.bar(x+0.5*width, data_to_cam, width=width, bottom=data_to_cloud, label='Feedback (EaOT)')
plt.xticks(x, (2, 4, 6, 8))


plt.legend()
plt.show()
