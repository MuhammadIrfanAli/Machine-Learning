function [N] = Normal_Distribution(x,m,C)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    N = exp(-1/2*(x-m)*inv(C)*(x-m)')./sqrt(det(2*pi*C));
end

