function [ avg_P ] = purity_func_soft(y_vec,c0_index,c1_index)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
% Calculate Purity Measure.
N = 20;
h = @(p) 1+p*log2(p)+(1-p)*log2(1-p);

p0 = (2/N)*sum(y_vec(c0_index));
p1 = (2/N)*sum(y_vec(c1_index));
%p0 = (2/N)*sum(y_vec(1:10));
%p1 = (2/N)*sum(y_vec(11:20));

if isreal(h(p0))
    P_s = h(p0);
else
    P_s = 1;
end
P_w = h(p1);
avg_P = (P_s+P_w)./2;
end

