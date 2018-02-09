function [c0_index,c1_index] = k_means_cluster(c0,c1)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% x_vec : Feature Vector
% k : Total Number of Clusters to create.

c0_index = [];
c1_index = [];

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
%     figure();
%     plot(c0(1),c0(2),'r*');
%     hold on;
%     plot(c1(1),c1(2),'b*');
%     % SCATTER PLOT
%     
%     scatter(c0_points(:,1),c0_points(:,2),[],'r');
%     %hold on;
%     scatter(c1_points(:,1),c1_points(:,2),[],'b');
%     title('Scatter Plot of X_G vs X_R');
%     xlabel('x_{redness}');
%     ylabel('x_{greeness}');
%     hold off;
end

end

