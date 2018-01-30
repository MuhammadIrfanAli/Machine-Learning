# Required Python Packages
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
#import tensorflow as tf
#from keras.models import Sequential
"""from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline"""
#import plotly.graph_objs as go
#import plotly.plotly as py
#from plotly.graph_objs import
#py.sign_in('Your_ployly_username', 'API_key')


def mult_nomial_logr():
    """ Prepare the Data """
    # Load the Training Data
    data_path_x = "E:/Masters\Machine Learning Basic Principles/Data Analysis Project/train_data.csv";
    songs_data = pd.read_csv(data_path_x, header=None);
    #print (songs_data.shape);
    # Lad the True Labels.
    data_path_y = "E:/Masters\Machine Learning Basic Principles/Data Analysis Project/train_labels.csv";
    songs_target = pd.read_csv(data_path_y, header=None);
    #print (songs_target.shape);
    data_path_z = "E:/Masters\Machine Learning Basic Principles/Data Analysis Project/test_data.csv";
    test_data = pd.read_csv(data_path_z, header=None);

    """ Split Data into Training and Validation """
    train_x, test_x, train_y, test_y = train_test_split(songs_data, songs_target, train_size=0.7, random_state=0);
    #lr = linear_model.LogisticRegression();
    #lr.fit(train_x, train_y);

    #print ("Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, lr.predict(train_x)));
    #print ("Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_y, lr.predict(test_x)));
    """Train multinomial logistic regression model"""
    mul_lr = linear_model.LogisticRegression(multi_class='multinomial', max_iter=10, solver='netwon-cg').fit(train_x, train_y);
    print ("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)));
    print ("Multinomial Logistic regression Test Accuracy ::  ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)));

    """ Predict the Test Data """
    test_data_results = mul_lr.predict(test_data);
    """ Export the Test Data File """
    df = pd.DataFrame(test_data_results);
    df.to_csv('test_data_results.csv', index=False, header=False);

###############################################################################

if __name__ == "__main__":
    """ First Approach: Multinomial Logistic Regression """
    mult_nomial_logr();
    """ Second Approach: Keras Deep Learning """
    #keras_deep_learn();
