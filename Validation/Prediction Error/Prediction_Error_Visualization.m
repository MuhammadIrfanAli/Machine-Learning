clear all; clc; close all;
% Load the Data.
load('data.mat');
% Evaluate the w vector.
x = X_train;
y = y_train;
w = inv(x'*x)*x'*y

% Select randomly 10 data points..
a = 1;
b = 101;
N = 10; % Samples to pick...
 
 for i=1:100
     r = floor((b-a).*rand(N,1) + a);% creates 10 random values
     for j=1:length(r)
         xVal_rand(j,:) = X_validation(r(j),:);
         yVal_rand(j) = y_validation(r(j));
     end
% Calculate the Empirical Error.
 ERM(i) = 1./N*sum((yVal_rand' - xVal_rand*w).^2);
 end
histogram(ERM);
grid on;
xlabel('Empirical Risk from 100 Randomly Selected Validation Data Points');
ylabel('frequency');
title ('Histogram of the Prediction Error');
%%%% Plotting and Things....
% h = w'*x';
% h = h';
% x_axis = 0:1./(length(h)-1):1;
% x_axis = x_axis';
% 
% scatter(x_axis,y);
% hold on;
% c = linspace(1,10,length(x));
% scatter(x_axis,h,[],c,'filled');