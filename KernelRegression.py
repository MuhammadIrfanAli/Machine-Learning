from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat;
from numpy.linalg import inv


""" GET THE Left 100x100 Green intensity Pixels and stack them int the vector"""
def get_feature_vector():
    x_vec = [];

    for i in range(N): #FOR ALL IMAGES
        im = Image.open("MontBlanc{}.png".format(i+1));
        width, height = im.size;
        im_pixels = width*height;
        rgb_im = im.convert('RGB'); # Converts the image into RGB
        sum_g = 0;
        for row in range(width):
            for col in range(height):
                r,g,b = rgb_im.getpixel((row,col));
                sum_g = sum_g + g;
        x_vec.append(sum_g);

    x_vec = np.array(x_vec); """ Convert to np array"""
    return x_vec;

    """ Kernel Regression DISCALIMER: This function works for 1D x_vector and 1D y_vector"""

def kernel_regression(feature_set,output,sigma):

    x_vec = feature_set;
    y = output;
    vec_len = len(x_vec);

    #x_len,y_len = x_vec.shape;
    kernel_mat = np.zeros((vec_len,vec_len)); # INIT
    norm_kernel = np.zeros((vec_len,vec_len)); # INIT
    row_weights = [];

    for i in range(N):
        for j in range(N):
            x_mu = x_vec[i] - x_vec[j];
            kernel_mat[i,j] = np.exp(-1/(2*sigma**2)*(x_mu)**2);

    row_weights = sum( [kernel_mat[i,:] for i in range(N)]);

    for i in range(N):
        for j in range(N):
            norm_kernel[i,j] = kernel_mat[i,j]/row_weights[i];
    h_x = np.dot(y,norm_kernel);
    return h_x;

""" ..................................MAIN............................................"""

if __name__ == '__main__':

    N = 10; # sample size;
    y = [211,271,121,31,341,401,241,181,301,301];
    y = np.array(y);
    sigma = [1,5,10];
    x_vec = get_feature_vector();

    h_x1 = kernel_regression(x_vec,y,sigma[0]); """ Returns Hypothesis Map """
    h_x5 = kernel_regression(x_vec,y,sigma[1]); """ Returns Hypothesis Map """
    h_x10 = kernel_regression(x_vec,y,sigma[2]); """ Returns Hypothesis Map """

    """ Order Indexes for Plots..."""
    indexs_to_order_by = x_vec.argsort();
    x_ordered = x_vec[indexs_to_order_by];
    y1_ordered = h_x1[indexs_to_order_by];
    y5_ordered = h_x5[indexs_to_order_by];
    y10_ordered = h_x10[indexs_to_order_by];
    """ PLOT """
    fig = plt.figure();
    fig.suptitle('Kernel Regression Plot', fontsize=16)
    plt.xlabel('Greenness', fontsize=14)
    plt.ylabel('Labelled Data(time in Minutes of images)', fontsize=14);

    plt.plot(x_ordered,y1_ordered,'--',label='KernelRegression with sigma=1');
    plt.plot(x_ordered,y5_ordered,'-.',label='KernelRegression with sigma=5');
    plt.plot(x_ordered,y10_ordered,':',label='KernelRegression with sigma=10');
    colors = np.random.rand(N);
    plt.scatter(x_vec,y, c = colors,label='Training Data');
    plt.legend(bbox_to_anchor=(1.00, 1), loc=2, borderaxespad=0.);
    plt.legend(loc = 'best');
    plt.show();
