from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat;
from numpy.linalg import inv
from scipy import misc
import random
import math
from sklearn.model_selection import KFold


def get_feature_vector(_size):
    N = 20;
    #r = random.randint(0,N);

    #x_vec = np.empty([]);
    x_vec = np.zeros((N,3*_size**2));
    #sq_dim = _size;

    x = [];
    for i in range(N):
        x = [];
        if (i < 10):
            im = Image.open("images/autumn{}.jpg".format(i+1));
        else:
            im = Image.open("images/winter{}.jpg".format((i-10)+1));
        rgb_im = im.convert('RGB');

        for row in range(_size):
            for col in range(_size):
                r,g,b = rgb_im.getpixel((row,col));
                x.append(r/255);
                x.append(g/255);
                x.append(b/255);
        x_vec[i,:]= x;
    #print (x_vec);
    return x_vec;

""" #####################################################"""

""" logistic_gradient_descent returns w_{opt}"""
def logistic_gradient_descent(x_vec,y_vec,_size):
    N = 20;
    w = np.zeros((3*_size**2,1));
    w_opt = np.zeros((3*_size**2,1));
    a = 1e-5; """ Learning Rate """
    yT = np.transpose(y_vec);
    xT = np.transpose(x_vec);
    error_margin = 0.0001;
    e = 1e20;
    _iter = 0;
    min_error = e;
    #for _iter in range(1000):
    while(abs(e) > error_margin):
        if (_iter > 1000):
            break;
        _num = math.exp(-np.dot(yT,np.dot(x_vec,w)));
        _den = 1 + math.exp(-np.dot(yT,np.dot(x_vec,w)));
        grad_w = -(5/N)*(_num/_den)*np.dot(xT,y_vec);
        w = w - a*grad_w;
        """ Computer Error """
        e = math.log(1 + math.exp(-np.dot(yT,np.dot(x_vec,w))));

        if (min_error > e):
            min_error = e;
            w_opt = w;
        #print (e);
        _iter = _iter + 1;
        #print(_iter);
    #print (w.shape);
    return (w_opt);

""" #####################################################"""
def _hypothesis(x_vec,w_opt,N):

    h_w = [];
    scores = [];
    h_w = (1/(1+np.exp(-np.dot(x_vec,w_opt))));
    #print ("test H is : ", h_temp);
    for i in range(N):
        if (h_w[i]) < 0.5:
            scores.append(-1);
        else:
            scores.append(1);

    """
    for i in range(N):
        try:
            #print ("_hypothesis values are:",1/(1+math.exp(-np.dot(x_vec[i,:],w_opt))));
            h_w.append(1/(1+math.exp(-np.dot(x_vec[i,:],w_opt))));
        except OverflowError:
            h_w.append(float(0));
            #h_w.append(float('inf'));
        if (h_w[i]) < 0.5:
            scores.append(-1);
        else:
            scores.append(1);

    h = np.array(h_w);
    scores = np.array(scores);
    h.shape = (N,1);
    scores.shape = (N,1);
    #print ("Hypothesis is",h);
    #return h_temp,scores;
    """
    h_w = np.array(h_w);
    scores = np.array(scores);
    h_w.shape = (N,1);
    scores.shape = (N,1);
    return h_w,scores;

""" #####################################################"""
""" returns avg training and testing error """
def k_cross_validation(x_vec,y_vec,k_splits,size):

    kf = KFold(n_splits=k_splits,random_state=None, shuffle=True);

    test_error_array = [];
    train_error_array = [];
    test_error = 0;
    train_error = 0;
    count = 0;
    for train_index, test_index in kf.split(x_vec):
        """ Make Training and Testing Data """
        #print ("im_dim is: ",size);
        #count = count + 1;
        #print ("Current Iteration is: ",count);
        #print ("Train Indices are ",train_index);
        #print ("Test Indices are ",test_index);
        x_train = x_vec[train_index, :];
        y_train = y_vec[train_index, :];
        x_test = x_vec[test_index,:];
        y_test = y_vec[test_index,:];

        w_opt = logistic_gradient_descent(x_train,y_train,size);
        h_w_train,scores_train = _hypothesis(x_train,w_opt,len(train_index));
        h_w_test,scores_test = _hypothesis(x_test,w_opt,len(test_index));
        #Calculate Validation Error
        #test_error = 1/(len(test_index))* sum([ (y_test[i] - 1/(1+math.exp(-np.dot(x_test[i],w_opt))))**2 for i in range(len(test_index))]);
        test_error = 1/(2*len(test_index))*np.dot(np.transpose(y_test - h_w_test),(y_test - h_w_test));
        test_error_array.append(test_error[0,0]);
        #print ("True Label Y_vec Train is:",y_train);
        #print ("Results of Trainings are:",result_train);
        #print ("True Label Y_vec for Test is :",y_test);
        #print ("Results of Testing are:",result_test);
            #print ("Test Results for Image Dimensions{} are".format(size),result_test);
            #print ("Training results for Image Dimensions{} are".format(size),result_train);
        #Calculate Training Error
        #train_error = 1/(len(train_index))* sum([ (y_train[i] - 1/(1+math.exp(-np.dot(x_train[i],w_opt))))**2 for i in range(len(train_index))]);
        train_error = 1/(2*len(train_index))*np.dot(np.transpose(y_train - h_w_train),(y_train - h_w_train));
        train_error_array.append(train_error[0,0]);

    return np.mean(train_error_array),np.mean(test_error_array);

""" ######################################################################### """
if __name__ == '__main__':

    N = 20; #Sample Size
    K = 5; #K-Fold Value.

    a1 = [1]*10; a2 = [-1]*10;
    y_vec = np.append(a1,a2);
    y_vec.shape = (20,1); """ force column shape to y_vec """

    #print ("y_vec is:",y_vec);
    im_dim = [1,10,20,50,100,200];
    im_dim2 = [1,10];
    E_train = [];
    E_test = [];
    e =[];

    #_kf = KFold(n_splits=K,random_state=None, shuffle=False);
    #[train_index, test_index] = _kf.split(x_vec);

    for i in range(len(im_dim)):
        size = im_dim[i];
        print ("im_dim is: ",size);
        x_vec = get_feature_vector(size); #TESTING
        w_opt = logistic_gradient_descent(x_vec,y_vec,size);
        #h = (1/(1+np.exp(-np.dot(x_vec,w_opt))));
        #h,scores = _hypothesis(x_vec,w_opt,N);
        #print ("Scores are: ",scores);
        #print ("Hypothesis values are:",h);
        train_e, test_e = k_cross_validation(x_vec,y_vec,K,size);
        E_train.append(train_e);
        E_test.append(test_e);
        #e1 = 1/(N)*np.dot(np.transpose(y_vec - scores),(y_vec - scores));
        #print (e1);
        #print (e1);
        #e.append(e1[0,0]);
    #print (w_opt.shape);
    #print ("True Label:",np.transpose(y_vec));
    #print ("Results are:",results);
    """
    for i in range(len(im_dim)):
        size = im_dim[i];"""
        #""" make feature vector with green and red intensities """
        #x_vec = get_feature_vector(size);
        #print (x_vec.shape);"""
        #""" perform k cross validation to evaluate avg training and testing error. """
        #train_e, test_e = k_cross_validation(x_vec,y_vec,K,size);
        #E_train.append(train_e);
        #E_test.append(test_e);
    #print (E_train, E_test);"""
    """ Plot IT """

    fig = plt.figure();

    fig.suptitle('Training/Test Error vs. Model Complexity', fontsize=16);
    plt.xlabel('Model Complexity', fontsize=14)
    plt.ylabel('Error', fontsize=14);
    plt.plot(im_dim,E_train,label='Training_Error');
    plt.plot(im_dim,E_test,'--',label='Validation_Error');
    plt.legend(bbox_to_anchor=(1.00, 1), loc=2, borderaxespad=0.);
    plt.legend(loc = 'best');
    plt.show();
