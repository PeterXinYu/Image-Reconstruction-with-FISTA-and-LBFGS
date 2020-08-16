function [f,phi,x0,opts] = get_para_fista(damaged_img_vec,undamaged_img_vec,mask_vec,omg,b_vec)
Py = @(y) y(omg);
Ax = @(x) Py(idct(x));
AAx = @(x) dct(mask_vec .* idct(x));

% setup: phi and f 

opts.mu          = 0.01;
phi.obj     = @(x) opts.mu * norm(x,1); %phi是非光滑部分，f是光滑部分
phi.prox    = @(x,lambda) x - max(min(opts.mu * lambda,x),-opts.mu * lambda);

f.obj       = @(x) 0.5 * norm(Ax(x) - b_vec)^2; %f是光滑部分
% f.grad      = @(x) AAx(x) - dct(P' * b_vec); 
% %f的gradient, 此处P' * b就是damaged_img_vec
f.grad      = @(x) AAx(x) - dct(damaged_img_vec);

% L           = norm(A'*A); % L就是A的最大特征值
% L           = @(x) norm(AA(x)); % L就是A的最大特征值
L           = 1;

opts.tol    = 1e-6;
% opts.lambda = 1.99/L; %恒定lambda
opts.lambda = 1; %恒定lambda
opts.maxit  = 100000; % 最大循环次数
opts.backMax = 10000;

x0          = ones(size(undamaged_img_vec,1),1);

opts.L          = L;
opts.backtr     = true;
opts.restart    = 0;
end

