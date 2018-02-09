function [ P_avg ] = purity_func_hard(c0_index,c1_index)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
h = @(p) 1+p*log2(p)+(1-p)*log2(1-p);
N = 20; k=2;
[p0_s y0_s] = find(c0_index(1,:)<11);
p_s_arg = length(p0_s)/(N./k);
[p1_w y1_w] = find(c0_index(1,:)>10);
p_w_arg = length(p1_w)/(N./k);
P_s = h(p_s_arg);
P_w = h(p_w_arg);
P_avg = (1/k)*(P_s+P_w);
end

