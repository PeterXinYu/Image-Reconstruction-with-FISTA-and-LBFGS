function [f,phi,x0,opts] = get_para_fista(damaged_img_vec,undamaged_img_vec,mask_vec,omg,b_vec)
Py = @(y) y(omg);
Ax = @(x) Py(idct(x));
AAx = @(x) dct(mask_vec .* idct(x));

% setup: phi and f 

opts.mu          = 0.01;
phi.obj     = @(x) opts.mu * norm(x,1); %phi�Ƿǹ⻬���֣�f�ǹ⻬����
phi.prox    = @(x,lambda) x - max(min(opts.mu * lambda,x),-opts.mu * lambda);

f.obj       = @(x) 0.5 * norm(Ax(x) - b_vec)^2; %f�ǹ⻬����
% f.grad      = @(x) AAx(x) - dct(P' * b_vec); 
% %f��gradient, �˴�P' * b����damaged_img_vec
f.grad      = @(x) AAx(x) - dct(damaged_img_vec);

% L           = norm(A'*A); % L����A���������ֵ
% L           = @(x) norm(AA(x)); % L����A���������ֵ
L           = 1;

opts.tol    = 1e-6;
% opts.lambda = 1.99/L; %�㶨lambda
opts.lambda = 1; %�㶨lambda
opts.maxit  = 100000; % ���ѭ������
opts.backMax = 10000;

x0          = ones(size(undamaged_img_vec,1),1);

opts.L          = L;
opts.backtr     = true;
opts.restart    = 0;
end

