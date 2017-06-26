import matplotlib.pyplot as plt
import numpy as np

n = 100

for color in ['red','blue','green']:
    x,y=np.random.rand(2,n)
    scale=100*np.random.rand(n)
    plt.scatter(x,y,c=color,s=scale,label=color,alpha=0.6,edgecolors='black')

plt.title('Scatter')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()