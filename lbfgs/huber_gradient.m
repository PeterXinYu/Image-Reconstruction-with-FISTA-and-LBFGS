function [y] = huber_gradient(x,delta)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
%     function [out] = gradient(x,delta)
%         if abs(x) <= abs(delta)
%             out = x / delta;
%         elseif x > delta
%             out = 1;
%         elseif x < -delta
%             out = -1;
%         end
%     end

% y = zeros(size(x,1),1);
% for i = size(x,1)
%     if abs(x(i)) <= delta
%         y(i) = x(i) / delta;
%     elseif x(i) > delta
%         y(i) = 1;
%     elseif x(i) < -delta
%         y(i) = -1;
%     end
% %         x(i) = gradient(x(i),delta);
% end

% % % % % % % % % % % % % % % % % % % % % % 

y = x;
ind=(abs(x)<=delta);
y(ind)=x(ind)/delta;
y(~ind)=sign(x(~ind));

end

