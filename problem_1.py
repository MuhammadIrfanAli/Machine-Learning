from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

red_points = [];
green_points = [];


for i in range(1,8):
    im = Image.open("shot%d.jpg" %i);
    #im1 = Image.open('E:\Masters\Machine Learning Basic Principles\Homework Problems\HW1\shot % name,.jpg') % (i);

    #r, g, b = im.split(); # splits image into r g b values...
    xsize, ysize = im.size; # Size of Image
    sum_r = 0; sum_g = 0; sum_b = 0;
    rgb_im = im.convert('RGB'); # Converts the image into RGB

    for row in range(xsize):
        for col in range(ysize):
            r,g,b = rgb_im.getpixel((row,col));
            sum_r = sum_r + r;
            sum_g = sum_g + g;
            #sum_b = sum_b + b;
    red_points.append(sum_r);
    green_points.append(sum_g);


print ("Red Points are:");
print red_points;
print ("Green Points are:");
print green_points;


"""label Data"""
fig = plt.figure();
fig.suptitle('Scatter Plot', fontsize=16)
plt.xlabel('Redness', fontsize=14)
plt.ylabel('Greenness', fontsize=14)

N = 7
colors = np.random.rand(N)

#plt.legend(('label1', 'label2', 'label3'))
#plt.legend(['A simple line']);
#print green_points[1];
"""PLOT AND SHOW"""
#for k in range(1,8):
plt.scatter(red_points,green_points, c = colors, alpha= 2.0);

"""label2 = plt.scatter(green_points[2], red_points[2], c= colors[1], alpha= 2.0);
label3 = plt.scatter(green_points[3], red_points[3], c=colors[2], alpha= 2.0);
label4 = plt.scatter(green_points[4], red_points[4], c=colors[3], alpha= 2.0);
label5 = plt.scatter(green_points[5], red_points[5], c=colors[4], alpha= 2.0);
label6 = plt.scatter(green_points[6], red_points[6], c=colors[5], alpha= 2.0);
#label7 = plt.scatter(green_points[k], red_points[k], c=colors[6], alpha= 2.0);
plt.legend((label1,label2,label3,label4,label5,label6),
            ('Shot1', 'Shot 2', 'Shot3', 'Shot4', 'Shot5', 'Shot6'),
            scatterpoints=1, loc='upper right', ncol=3, fontsize=8);"""
plt.show();
