function [x,out] = lbfgs(f,x0,opts,undamaged_img_vec)

mn = size(x0,1); %Ϊ��ȷ��H��V�Ĵ�С
I = speye(mn,mn); %ά��Ϊmn * mn�ĵ�λ����

fprintf(1,'\n- - - L_BFGS\n');
fprintf(1,'ITER ; OBJ.VALUE ; RES.NORM ; PSNR\n');

x       = x0;
k       = 0;
% H_old   = I; %H0��Ϊ��λ����
r       = f.grad(x); %��ʼ��r����Ϊ�ݶ�
f_old   = f.obj(x); 

%����L-BFGS�ļ�����Ҫ�����ı��ʽ
% s       = @(x,x_old) x - x_old; %s�ı��ʽ��x(k+1) - x(k)
% y       = @(x,x_old) f.grad(x) - f.grad(x_old);%y�ı��ʽ���ݶ�(x(k+1)) - �ݶ�(x(k))
% rho     = @(x,x_old) 1 / (y(x,x_old)' * s(x,x_old)); %rho�ı��ʽ
% V       = @(x,x_old) I - rho(x,x_old) * y(x,x_old)' * s(x,x_old); %����V�ı��ʽ
% �ú����������ٶȻ����

% m = 1; %�����������Ƶ�ǰ�洢�������Եĸ���
% y_mat = zeros(mn,opts.m); %���m��y
% s_mat = zeros(mn,opts.m); %���m��s
% rho_mat= zeros(opts.m,1); %���m��rho
alpha_mat = zeros(opts.m,1); %���m��alpha�����alpha���ǲ���������������Ĳ���

y_mat = []; %���m��y
s_mat = []; %���m��s
rho_mat= []; %���m��rho

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
    
    alpha  = opts.s; %��ʼ��alpha����Ϊs=1
    x      = x_old - alpha * r; %�˴�д�Ӻţ���Ϊp=-r
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
            y_mat= [y_mat,y]; %����ϲ�
            %m / opts.mû��ȡ��1�Ļ����ͼ����棬�����˾͸���r
            s_mat = [s_mat,s];
            rho_mat = [rho_mat;rho];%����ϲ�
        else
            y_mat(:,1) = [];
            y_mat(:, opts.m) = y;
            s_mat(:,1) = [];
            s_mat(:, opts.m) = s; 
            rho_mat(1,:) = [];
            rho_mat(opts.m,:) = rho;
            
%             y_mat = [y_mat, y(x,x_old)]; %����ϲ�
%             y_mat = y_mat(:,2:opts.m + 1);
%             s_mat = [s_mat, s(x,x_old)];
%             s_mat = s_mat(:,2:opts.m + 1);
%             rho_mat = [rho_mat; rho(x,x_old)]; %����ϲ�
%             rho_mat = rho_mat(2:opts.m + 1);
        end
    end
    
%     if size(y_mat,2) == opts.m
    if true
%         H0 = s_mat(:,opts.m)' * y_mat(:,opts.m) / norm(y_mat(:,opts.m)) ^ 2; %�����㷨���ó�ʼ��H0
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
%         H0 = s_mat(:,size(s_mat,2))' * y_mat(:,size(y_mat,2)) / norm(y_mat(:,size(y_mat,2))) ^ 2; %�����㷨���ó�ʼ��H0
%         q  = f.grad(x);
%         r  = H0 * q;
%     end
    
    
%     if index == 5
%         index = 1;
%     end
    
    %��m == opts.m����˵�������ˣ���ʱ�����r
%     if m == opts.m
%         H0 = s_mat(:,m)' * y_mat(:,m) / rho_mat(m); %�����㷨���ó�ʼ��H0
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

