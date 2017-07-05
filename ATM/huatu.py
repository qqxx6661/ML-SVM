import matplotlib.pyplot as plt
import numpy as np

i = 0
for color in ['blue','orange','red']:
    data = []
    labels = []
    with open("ATM/ATM_huatu3.txt") as file:
        for line in file:
            tokens = line.strip().split(' ')
            try:
                data.append(tokens[i+1])
                labels.append(tokens[i])

            except IndexError:
                continue
        print(data)
        print(labels)
    i = i + 2
    x = np.array(data)
    y = np.array(labels)
    plt.scatter(x,y,c=color,s=50,label=color,alpha=1,edgecolors='black')

i = 0
for color in ['green','blue','orange','red']:
    data = []
    labels = []
    with open("ATM/ATM_huatu2.txt") as file:
        for line in file:
            tokens = line.strip().split(' ')
            try:
                data.append(tokens[i+1])

                labels.append(tokens[i])

            except IndexError:
                continue
        print(data)
        print(labels)
    i = i + 2
    x = np.array(data)
    y = np.array(labels)
    plt.scatter(x,y,c=color,s=50,label=color,alpha=0.6,edgecolors='white')

# plt.title('3、4月份预测数据图')
plt.xlabel('Response Delay(ms)')
plt.ylabel('Success Rate(%)')
plt.legend()
plt.grid(True)
plt.show()
