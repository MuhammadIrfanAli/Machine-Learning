from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat;
from numpy.linalg import inv


N = 10; # sample size;
x_vector = np.ones((10,2));
y_label = np.array([211,271,121,31,341,401,241,181,301,301]);
y_vector = np.transpose(y_label);

for i in range(N):
    im = Image.open("MontBlanc{}.png".format(i+1));
    xsize, ysize = im.size; # Size of Image
    total_pixels = xsize*ysize;
    sum_g = 0; sum_r = 0; sum_b = 0;
    rgb_im = im.convert('RGB'); # Converts the image into RGB

    for row in range(xsize):
        for col in range(ysize):
            r,g,b = rgb_im.getpixel((row,col));
            #g = g/(r+g+b);
            sum_g = sum_g + g;
            #sum_r = sum_r + r;
            #sum_b = sum_b + b;
    x_vector[i,0] = sum_g/total_pixels;
    #g = sum_g/(sum_g+sum_r+sum_b);
print (x_vector);

xT = np.transpose(x_vector);
xTx = np.dot(xT,x_vector);
xTx_inv = inv(xTx);
print ("Inverse Matrix is: ");
print (xTx_inv);
xTy = np.dot(xT,y_vector);
print ("xTy Vector is: ");
print (xTy);

w_opt = np.dot(xTx_inv,xTy);
w1 = w_opt[0]; w0 = w_opt[1];
print (w1); print (w0);
x_range = np.array(range(60, 130));

h = eval('w1*x_range+w0');

fig = plt.figure();
fig.suptitle('Plot of Norm_x_g vs Label ', fontsize=16)
plt.xlabel('Normalized Green Points', fontsize=14)
plt.ylabel('Time After 07:00 AM in Minutes', fontsize=14);
plt.plot(x_range,h);
N = 10;
colors = np.random.rand(N)
plt.scatter(x_vector[:,0],y_label, c = colors, alpha= 2.0);
plt.show();

"""
a = np.arange(20).reshape((10,2));
r = np.ones((10,2));
for i in range(10):
    r[i,1] = i;
#a = np.zeros((10, 1));
#for i in range(10):
#    a = np.vstack(a,[i,a[i]]);
print (r);
"""


"""hour_clock = 12;
mint_clock = 1;
ref = 7;
print ((hour_clock-ref)*60+mint_clock);"""
