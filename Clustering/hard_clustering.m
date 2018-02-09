clc; clear all; close all;

%% Read Images and get x_g and x_r
N = 20; % total images.
k = 2;

for j=1:N
    if (j <= 10)
        RGB = imread(sprintf('images/summer%d.jpeg',j));
    elseif (j>10)
        RGB = imread(sprintf('images/winter%d.jpeg',j-10));
    end    
    [row,col,page]= size(RGB);
    dim = row*col;
    im_r = RGB(:,:,1);
    im_g = RGB(:,:,2);
    sum_col_r = sum(im_r);
    sum_r = sum(sum_col_r,2);
    sum_col_g = sum(im_g);
    sum_g = sum(sum_col_g);

    % stack values into feature vector X
    x_vec(j,:) = [sum_r./dim,sum_g./dim];
end
%x_vec = x_vec./255;

%x_Vec
figure();
scatter(x_vec(1:10,1),x_vec(1:10,2),'r');
hold on;
scatter(x_vec(11:20,1),x_vec(11:20,2),'b');
hold off;
%
% idx = kmeans(x_vec,2);
% [x11,y11]=find(idx(:,1)<2);
% [x22,y22]=find(idx(:,1)>1);
% figure();
% scatter(x_vec(x11,1),x_vec(x11,2),[],'r');
% hold on;
% scatter(x_vec(x22,1),x_vec(x22,2),[],'b');


% 
% figure();
% plot(c0(1),c0(2),'g*');
% hold on;
% plot(c1(1),c1(2),'g*');
% hold off;
y = ones(N,1)*-1; % Y contains to which cluster does the point belongs.
%% Calculate Distance of Each point from the Centroid.
M = linspace(1,50,50);
%% Define Purity Function.
P = [];

for m=1:length(M)
     

    for n=1:10 % Take 10 random samples.
        P_bar = [];
        % Initialize Random Centroids.
        r = my_rand_func(1,20,2); % Input Args: lower_limit, upper_limit,samples
        c0 = x_vec(r(1),:);
        c1 = x_vec(r(2),:);    
 
        for i=1:M(m)
            c0_index = [];
            c1_index = [];
 
            % Calculate distance and update Cluster Assignment.
            for j=1:20 % for total number of images.

                D0 = sum((x_vec(j,:) - c0) .^ 2);
                D1 = sum((x_vec(j,:) - c1) .^ 2);

                if (D0<D1)
                    y(j) = 0;
                    c0_index = [c0_index,j]; %save index of points 
                else
                    y(j) = 1;
                    c1_index = [c1_index,j]; %save index of points 
                end    

            end
            % Calculate Means of the new points...
            c0_points = x_vec(c0_index,:);
            c1_points = x_vec(c1_index,:);
            %Update c0 and c1.
            c0 = mean(c0_points);
            c1 = mean(c1_points);
            %figure();
    %         plot(c0(1),c0(2),'r*');
    %         hold on;
    %         plot(c1(1),c1(2),'b*');
    %         %     % SCATTER PLOT
    %         %     
    %         scatter(c0_points(:,1),c0_points(:,2),[],'r');
    %         %hold on;
    %         scatter(c1_points(:,1),c1_points(:,2),[],'b');
    %         title('Scatter Plot of X_G vs X_R');
    %         xlabel('x_{redness}');
    %         ylabel('x_{greeness}');
    %         hold off;
    
        end % for i=1:m.  
        P_avg = purity_func_hard(c0_index,c1_index);
        P_bar = [P_bar,P_avg];
    end
    P = [P,mean(P_bar)];
end
figure();
plot(M,P,'-o');
xlabel('Number of Iterations (M)');
ylabel('Average Purtiy (P)');
