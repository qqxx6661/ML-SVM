import numpy as np
data = []
labels = []
with open("train/train.csv") as file:
    for line in file:
        tokens = line.strip().split(',')
        data.append([float(tk) for tk in tokens[1:4]])
        # print(data)
        labels.append(tokens[0])

X = np.array(data)
y = np.array(labels)

with open("train/sample2_2.csv") as file:
    for line in file:
        tokens = line.strip().split(',')
        data.append([float(tk) for tk in tokens[1:4]])
        # print(data)
        labels.append(tokens[0])

X_2 = np.array(data)
y_2 = np.array(labels)

X_3 = X_2 + X
y_3 = y_2 + y
