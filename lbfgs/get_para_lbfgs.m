function [f,x0,opts] = get_para_lbfgs(damaged_img_vec,undamaged_img_vec,mask_vec,omg,b_vec)
Py = @(y) y(omg);
Ax = @(x) Py(idct(x));
AAx = @(x) dct(mask_vec .* idct(x));

x0          = ones(size(damaged_img_vec,1),1);

mu          = 0.005;
delta        = 0.05;
L           = 2;

f.obj       = @(x) 0.5 * norm(Ax(x) - b_vec)^2 + mu * huber_function(x,delta); %f是光滑部分
% f.grad      = @(x) AAx(x) - dct(P' * b_vec); 
% %f的gradient, 此处P' * b就是damaged_img_vec
f.grad      = @(x) AAx(x) - dct(damaged_img_vec) +  mu * huber_gradient(x,delta);
f.sigma     = 0.1;

opts.tol    = 1e-6;
% opts.lambda = 1.99/L; %恒定lambda
opts.lambda = 1; %恒定lambda
opts.maxit  = 1000; % 最大循环次数
opts.backMax = 100;
opts.m      = 5; %l-bfgs一次存储的个数

opts.L          = L;

% backtracking要用到的参数
opts.backtr     = true;
opts.s          = 1;
opts.sigma      = 0.5;
opts.gmma       = 0.1;

opts.restart    = 0;
end

