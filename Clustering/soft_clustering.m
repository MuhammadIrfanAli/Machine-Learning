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
    im_size = row*col;
    im_r = RGB(:,:,1);
    im_g = RGB(:,:,2);
    sum_col_r = sum(im_r);
    sum_r = sum(sum_col_r,2);
    sum_col_g = sum(im_g);
    sum_g = sum(sum_col_g);

    % stack values into feature vector X
    x_vec(j,:) = [sum_r./im_size,sum_g./im_size];
end
% Normalize the x_vec
x_vec = x_vec./255;
y_vec = ones(20,1)*-1; %x_m = mean(x_vec); %x_s = std(x_vec); %x_vec = (x_vec - x_m)./x_s

%% 
M = linspace(1,100,100);
P_M = []; % Final vector that contains Average Purity against every M.
P_iter = [];

r = my_rand_func(1,20,2); % Input Args: lower_limit, upper_limit,samples
    % Intitial Guesses for m0,m1,C0,C1
    m0 = x_vec(r(1),:);
    m1 = x_vec(r(2),:);
    cov_0 = eye(2);
    cov_1 = eye(2);

for m = 1:length(M)
    
%     r = my_rand_func(1,20,2); % Input Args: lower_limit, upper_limit,samples
%     % Intitial Guesses for m0,m1,C0,C1
%     m0 = x_vec(r(1),:);
%     m1 = x_vec(r(2),:);
%     cov_0 = eye(2);
%     cov_1 = eye(2);
    
    for i=1:M(m)
            
        cluster0_index = [];
        cluster1_index = [];

        % Prob of point x(i) belonging to dataset Cluster0.
        for j=1:N
            N0_temp = Normal_Distribution(x_vec(j,:),m0,cov_0);
            N1_temp = Normal_Distribution(x_vec(j,:),m1,cov_1);
            y_vec(j) = N0_temp/(N0_temp+N1_temp);

            if (y_vec(j) > 1-y_vec(j))
                cluster0_index = [cluster0_index,j];
            else
                cluster1_index = [cluster1_index,j];
            end

        end
        % <UPDATE PARAMETERS>
        prob_size_N0 = sum(y_vec);
        prob_size_N1 = N - prob_size_N0;

        m0 = (1/prob_size_N0)*sum((y_vec.*x_vec));%m_0
        m1 = (1/prob_size_N1)*sum(((1-y_vec).*x_vec));%m_1
        temp0 = 0;
        temp1 = 0;
        for k=1:N
            temp0 = temp0 + y_vec(k)*(x_vec(k,:)-m0)'*(x_vec(k,:)-m0);
            temp1 = temp1 + (1-y_vec(k))*(x_vec(k,:)-m1)'*(x_vec(k,:)-m1);
        end
        cov_0 = (1/prob_size_N0)*temp0;
        cov_1 = (1/prob_size_N1)*temp1;

        %cluster0_points = x_vec(cluster0_index,:);
        %cluster1_points = x_vec(cluster1_index,:);
        P_cur = purity_func_soft(y_vec,cluster0_index,cluster1_index);    
        P_iter = [P_iter,P_cur];
        end % for i=1:M(m)
        
        %P_cur = purity_func_soft(y_vec,cluster0_index,cluster1_index);
        %P_iter = [P_iter,P_cur];
        P_M = [P_M,mean(P_iter)];
%     if (i>20)
%         figure();
% 
%         scatter(cluster0_points(:,1),cluster0_points(:,2),[],'r');
%         hold on;
%         scatter(cluster1_points(:,1),cluster1_points(:,2),[],'b');
%         hold off;
%     end
    %end %for i=1:10
        %P_M = [P_M,mean(P_iter)];%after 10 iterations, we have good estimate of P@M.
end % for m=1:length(M)

plot(M,P_M,'-or');
xlabel('Number of Iterations');
ylabel('Average Purity');
%title('With Random Cluster Centroids');