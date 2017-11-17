from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

green_points = [];
norm_x_g = [];
for i in range(1,11):
    im = Image.open("MontBlanc{}.png".format(i));
    xsize, ysize = im.size; # Size of Image
    total_pixels = xsize*ysize;
    #print ("Size of image is{}" .format(im.size));
    #print (xsize); print (ysize);print (xsize*ysize);
    sum_g = 0;
    rgb_im = im.convert('RGB'); # Converts the image into RGB

    for row in range(xsize):
        for col in range(ysize):
            r,g,b = rgb_im.getpixel((row,col));
            sum_g = sum_g + g;
    norm_x_g.append(sum_g//total_pixels);

#print ("Green Points are:");
#print (green_points);
label = [211,271,121,31,341,401,241,181,301,301];


"""a = np.array([[1,2],[2,3]]);
a_inv = inv(a);
print (a_inv);"""




""" GET IMAGE LABEL"""
"""hour_clock = 12;
mint_clock = 1;
ref = 7;
print ((hour_clock-ref)*60+mint_clock);
fig = plt.figure();
fig.suptitle('Plot of Norm_x_g vs Label ', fontsize=16)
plt.xlabel('Normalized Green Points', fontsize=14)
plt.ylabel('Time After 07:00 AM in Minutes', fontsize=14);

N = 10;
colors = np.random.rand(N)
plt.scatter(norm_x_g,label, c = colors, alpha= 2.0);
plt.show();

#plt.legend(('label1', 'label2', 'label3'))
#plt.legend(['A simple line']);
#print green_points[1];
"""
