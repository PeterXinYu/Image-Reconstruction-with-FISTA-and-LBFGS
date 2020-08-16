function [x,out] = lbfgs(f,x0,opts,undamaged_img_vec)

mn = size(x0,1); %为了确定H和V的大小
I = speye(mn,mn); %维度为mn * mn的单位矩阵

fprintf(1,'\n- - - L_BFGS\n');
fprintf(1,'ITER ; OBJ.VALUE ; RES.NORM ; PSNR\n');

x       = x0;
k       = 0;
% H_old   = I; %H0设为单位矩阵
r       = f.grad(x); %初始的r设置为梯度
f_old   = f.obj(x); 

%设置L-BFGS的几个重要参数的表达式
% s       = @(x,x_old) x - x_old; %s的表达式，x(k+1) - x(k)
% y       = @(x,x_old) f.grad(x) - f.grad(x_old);%y的表达式，梯度(x(k+1)) - 梯度(x(k))
% rho     = @(x,x_old) 1 / (y(x,x_old)' * s(x,x_old)); %rho的表达式
% V       = @(x,x_old) I - rho(x,x_old) * y(x,x_old)' * s(x,x_old); %矩阵V的表达式
% 用函数来计算速度会很慢

% m = 1; %用来计数，计当前存储的向量对的个数
% y_mat = zeros(mn,opts.m); %存放m个y
% s_mat = zeros(mn,opts.m); %存放m个s
% rho_mat= zeros(opts.m,1); %存放m个rho
alpha_mat = zeros(opts.m,1); %存放m个alpha，这个alpha不是步长，是两步法里的参数

y_mat = []; %存放m个y
s_mat = []; %存放m个s
rho_mat= []; %存放m个rho

out.trace = zeros(opts.maxit,1);
out.psnr = zeros(opts.maxit,1);

for iter = 1:opts.maxit
    
    res = norm(f.grad(x));
    k   = 1;
    if  res<= opts.tol
        break
    end
    
    x_old  = x;
    g_old  = f.grad(x_old);
    
%     p      = -r;
%     if m <= opts.m
%         r = f.grad(x_old);
%     end
    
    alpha  = opts.s; %初始的alpha设置为s=1
    x      = x_old - alpha * r; %此处写加号，因为p=-r
    f_new  = f.obj(x);
    g      = f.grad(x);
    
    psnr_value = psnr(undamaged_img_vec, x);
    out.trace(iter) = f_new; 
    out.psnr(iter) = psnr_value;
    
    fprintf(1,'[%4i] ; %1.6f ; %1.6e; %2.4f \n',iter,f_old,res,psnr_value);
    
    while f_new - f_old > - alpha * opts.gmma * (g' * r) && k <= opts.backMax
        alpha = alpha * opts.sigma;
        x     = x_old - alpha * r;
        f_new = f.obj(x);
        k     = k + 1;
    end
    
    f_old = f_new;
    
    s = x - x_old;
    y = g - g_old;
    rho = 1 / (s' * y);
    
    if s' * y > 1e-14
%     if true
        if size(y_mat,2) ~= opts.m
            y_mat= [y_mat,y]; %横向合并
            %m / opts.m没有取到1的话，就继续存，存满了就更新r
            s_mat = [s_mat,s];
            rho_mat = [rho_mat;rho];%纵向合并
        else
            y_mat(:,1) = [];
            y_mat(:, opts.m) = y;
            s_mat(:,1) = [];
            s_mat(:, opts.m) = s; 
            rho_mat(1,:) = [];
            rho_mat(opts.m,:) = rho;
            
%             y_mat = [y_mat, y(x,x_old)]; %横向合并
%             y_mat = y_mat(:,2:opts.m + 1);
%             s_mat = [s_mat, s(x,x_old)];
%             s_mat = s_mat(:,2:opts.m + 1);
%             rho_mat = [rho_mat; rho(x,x_old)]; %纵向合并
%             rho_mat = rho_mat(2:opts.m + 1);
        end
    end
    
%     if size(y_mat,2) == opts.m
    if true
%         H0 = s_mat(:,opts.m)' * y_mat(:,opts.m) / norm(y_mat(:,opts.m)) ^ 2; %按照算法设置初始的H0
        H0 = s' * y / norm(y) ^ 2 * I;
        q = g;
        for i = size(y_mat,2):-1:1
            alpha_mat(i) = rho_mat(i) * s_mat(:,i)' * q;
            q = q - alpha_mat(i) * y_mat(:,i);
        end
        r = H0 * q;
        for i = 1:size(y_mat,2)
            beta = rho_mat(i) * y_mat(:,i)' * r;
            r = r + (alpha_mat(i) - beta) * s_mat(:,i);
        end
    end
    
%     if m > 1 && m <=opts.m
%         H0 = s_mat(:,size(s_mat,2))' * y_mat(:,size(y_mat,2)) / norm(y_mat(:,size(y_mat,2))) ^ 2; %按照算法设置初始的H0
%         q  = f.grad(x);
%         r  = H0 * q;
%     end
    
    
%     if index == 5
%         index = 1;
%     end
    
    %当m == opts.m，就说明存满了，这时候更新r
%     if m == opts.m
%         H0 = s_mat(:,m)' * y_mat(:,m) / rho_mat(m); %按照算法设置初始的H0
%         q = f.grad(x);
%         for i = m:-1:1
%             alpha_mat(i) = rho_mat(m) * s_mat(:,m)' * q;
%             q = q - alpha_mat(i) * y_mat(:,m);
%         end
%         r = H0 * q;
%         for i = m:-1:1
%             beta = rho_mat(i) * y_mat(:,m)' * r;
%             r = r + (alpha_mat(i) - beta) * s_mat(:,i);
%             
%         end
%         disp(size(r))
% 
%         m = 0;
%         break
%     end
% % % % % % % % % % % % % 
    

    
% % % % % % % % % % % % % %     
%     disp(size(r))

%     break

    
    
end

out.trace = out.trace(1:iter-1);
out.psnr = out.psnr(1:iter-1);
    
end

