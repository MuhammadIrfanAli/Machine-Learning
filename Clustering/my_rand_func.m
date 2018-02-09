function [ r ] = my_rand_func(lower_num,upper_num,samples)
% my_rand_func(lower_limit, upper_limit,number_of_samples)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
while (1)
    r = floor((upper_num-lower_num).*rand(samples,1) + lower_num);% creates 2 random values
    if (r(1)~=r(2))
        break;
    end
end

end

