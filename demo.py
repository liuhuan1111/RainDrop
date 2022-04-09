import matplotlib.pyplot as plt
import numpy as np

out1 = np.loadtxt("./MSELoss_list.txt")
out2 = np.loadtxt("./MAELoss_list.txt")
out3 = np.loadtxt("./HuberLoss_list.txt")

plt.figure(dpi=500)
x = range(0, 2250)
y1 = out1
y2 = out2
y3 = out3
plt.plot(x, y1, label='MSELoss', lw=0.3)
plt.plot(x, y2, label='MAELoss', lw=0.3)
plt.plot(x, y3, label='HuberLoss', lw=0.3)
plt.legend()
plt.ylabel('loss')
plt.xlabel('times')
plt.savefig('./loss.jpg')
plt.show()
