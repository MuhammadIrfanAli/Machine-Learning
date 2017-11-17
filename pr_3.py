from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat;
from numpy.linalg import inv


N = 10; # sample size;
x_vector = np.ones((10,2));
y_label = np.array([211,271,121,31,341,401,241,181,301,301]);
y_vector = np.transpose(y_label);
lambda_vec = [2,5];

for i in range(N):
    im = Image.open("MontBlanc{}.png".format(i+1));
    xsize, ysize = im.size; # Size of Image
    total_pixels = xsize*ysize;
    sum_g = 0;
    rgb_im = im.convert('RGB'); # Converts the image into RGB

    for row in range(xsize):
        for col in range(ysize):
            r,g,b = rgb_im.getpixel((row,col));
            sum_g = sum_g + g;#/(r+g+b));
    x_vector[i,0] = sum_g/total_pixels;

#_mean = np.mean(x_vector[:,0]);
#_std_dev = np.std(x_vector[:,0]);
#x_vector[:,0] = (x_vector[:,0] - _mean)/_std_dev;
#print (x_vector);
""" w_opt = (X^{T}X+lambda*I)^-1 * X^{T}Y """

xT = np.transpose(x_vector);
xTy = np.dot(xT,y_vector);
xTx_inv_0 = inv(xTx); #lambda_vec=5

xTx = np.dot(xT,x_vector);
xTx_inv = inv(xTx+np.dot(lambda_vec[0],np.identity(2))); #lambda_vec=2
xTx_inv_1 = inv(xTx+np.dot(lambda_vec[1],np.identity(2))); #lambda_vec=5




w_opt = np.dot(xTx_inv,xTy);
w1 = w_opt[0];
w0 = w_opt[1];
x_range = np.linspace(np.min(x_vector[:,0]),np.max(x_vector[:,0]),100);
h1 = eval('w1*x_range+w0');

w_opt_1 = np.dot(xTx_inv_1,xTy);
w11 = w_opt_1[0]; w00 = w_opt_1[1];
h2 = eval('w11*x_range+w00');

w_opt_0 = np.dot(xTx_inv_0,xTy);
w_1 = w_opt_0[0]; w_0 = w_opt_0[1];
h0 = eval('w_1*x_range+w_0');


fig = plt.figure();
fig.suptitle('Plot of Norm_x_g vs Label ', fontsize=16)
plt.xlabel('Standardized Green Points', fontsize=14)
plt.ylabel('Time After 07:00 AM in Minutes', fontsize=14);

plt.plot(x_range,h1,'--',label='Lambda = 2');
plt.plot(x_range,h2,'-.',label='Lambda = 5');
#plt.plot(x_range,h0,':',label='Lambda = 0');

plt.legend(loc = 'best');
#plt.legend(handles=[line2, line5])
N = 10;
colors = np.random.rand(N)
plt.scatter(x_vector[:,0],y_label, c = colors, alpha= 2.0);
plt.show();
