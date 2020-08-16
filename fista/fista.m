function [x,out] = fista(f,phi,x0,opts,undamaged_img_vec)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INNER FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function [z,l,f_new,res] = backTracking(yy,gg,l,eta)
        k           = 1;
        z           = phi.prox(yy-gg/l,1/l); 
        r           = z - yy;
        f_old       = f.obj(yy);
        f_new       = f.obj(z);
        
        while f_new - f_old > gg'*r + norm(r)^2/(2*l) && k <= opts.backMax
            k       = k+1;
            l       = l/eta;
            
            z       = phi.prox(yy-gg/l,1/l); 
            r       = z - yy;	
            f_new   = f.obj(z);
        end
        
        res         = norm(r);
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARAMETERS & INITIALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x       = x0;
x_old   = x;

t       = 1;
t_old   = 1;

if ~opts.backtr
    L   = opts.L;
else
    L   = opts.L; eta = 0.5;
end

fprintf(1,'\n- - - fista\n');
fprintf(1,'ITER ; OBJ.VALUE ; RES.NORM ; L ; PSNR\n');

out.trace = zeros(opts.maxit,1);
out.psnr = zeros(opts.maxit,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for iter = 1:opts.maxit
    
    beta        = (t_old - 1)/t; 
    y           = x + beta*(x-x_old);
    x_old       = x;
    
    g           = f.grad(y);
      
    if ~opts.backtr
        x       = phi.prox(y-g/L,1/L); %算子，就是算法7的步骤
        res     = norm(x-y);
        f_obj   = f.obj(x);
    else 
        [x,L,f_obj,res]  = backTracking(y,g,L,eta);
    end
    
    psi_obj     = f_obj + phi.obj(x);
    psnr_value = psnr(undamaged_img_vec, x);
    fprintf(1,'[%4i] ; %1.6f ; %1.6e ; %2.4f; %2.4f\n',iter,psi_obj,res,L,psnr_value);
    
    out.trace(iter) = psi_obj;
    out.psnr(iter)  = psnr_value;
    
    if res <= opts.tol
        break
    end
         
    t_old   = t;
    t       = (1+sqrt(1+4*t^2))/2; 
    
    if opts.restart > 0
        if mod(iter,opts.restart) == 0
            t_old = 1; t = 1;
        end
    end
end

out.trace = out.trace(1:iter);
out.psnr = out.psnr(1:iter);

end

