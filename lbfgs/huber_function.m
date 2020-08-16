function [out] = huber_function(x,delta)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
out = 0;
for i = 1:size(x,1)
    if abs(x(i)) <= delta
        out = out + x(i) ^ 2 / (2 * delta);
    else
        out = out + abs(x(i)) - delta / 2;
    end
end
end

