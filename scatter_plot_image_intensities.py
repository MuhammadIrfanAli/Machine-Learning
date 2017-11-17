import numpy as np
import matplotlib.pyplot as plt

mean = [0]*10;
cov = [ [1,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,1] ];
z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8, z_9, z_10 = np.random.multivariate_normal(mean, cov, 100).T;

u = np.array([1,0,0,0,0,0,0,0,0,0]);
#u_T = np.transpose(u);
v = np.array([0.9,0.1,0,0,0,0,0,0,0,0]);
#test = [9/10,1/10,0,0,0,0,0,0,0,0];

""" Declare the resulting arrays"""
x_1 = [];
x_2 = [];

for i in range(0,100):
    z = np.array([ z_1[i], z_2[i], z_3[i], z_4[i], z_5[i], z_6[i], z_7[i], z_8[i], z_9[i], z_10[i] ]);
    z_T = np.transpose(z);
    #print z_T;
    x_1.append(np.dot(u,z));
    x_2.append(np.dot(v,z));

N = 100;
colors = np.random.rand(N);

fig = plt.figure();
fig.suptitle("Scatter Plot of x_1 vs x_2", fontsize = 16);
plt.xlabel("x_1", fontsize = 14);
plt.ylabel("x_2", fontsize = 16);

plt.scatter(x_1,x_2, c=colors, alpha = 2.0);
plt.show();

"""
mean = [1, 10, 100]
cov = [[1,1,1], [1,1,1], [1,1,1]]
x, y, z = np.random.multivariate_normal(mean, cov, 100).T
print x
print y
print z
"""
"""
mean = 0;
std_dev = 1;
#x_1 = np.random.normal(mean, std_dev, 1000);
#np.std(s, ddof=1)

count, bins, ignored = plt.hist(x_2, 30, normed=True);
plt.plot(bins, 1/(std_dev * np.sqrt(2 * np.pi)) * np.exp( - (bins - mean)**2 / (2 * std_dev**2) ), linewidth=2, color='r');
plt.show();"""
