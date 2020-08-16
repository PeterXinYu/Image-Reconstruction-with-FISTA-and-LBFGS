addpath lbfgs
addpath fista

[damaged_img_vec,undamaged_img_vec,mask_vec,omg,b_vec] = get_img_para('../test_images/640_640_lion_grey.png','../test_masks/640_640_handwriting.png');
[f,phi,x0,opts] = get_para_fista(damaged_img_vec,undamaged_img_vec,mask_vec,omg,b_vec);

opts.mu = 0.01;
opts.delta = 0.1;
opts.m = 5;
opts.maxit = 10000;
opts.tol = 1e-2;

tic
[x,out1]    = fista(f,phi,x0,opts,undamaged_img_vec);
psnr_value  = psnr(undamaged_img_vec, x);
t           = toc;
fprintf(1,'--- fista: [ITER; OBJ]: [%4i; %1.2e]; TIME: %2.2f sec; PSNR: %2.2f \n',length(out1.trace),out1.trace(end),t,psnr_value);

[f,x0,opts] = get_para_lbfgs(damaged_img_vec,undamaged_img_vec,mask_vec,omg,b_vec);
opts.tol = 1e-2;

tic
[x,out2]    = lbfgs(f,x0,opts,undamaged_img_vec);
psnr_value  = psnr(undamaged_img_vec, x);
t           = toc;
fprintf(1,'--- L-BFGS: [ITER; OBJ]: [%4i; %1.2e]; TIME: %2.2f sec; PSNR: %2.2f\n',length(out2.trace),out2.trace(end),t, psnr_value);


figure;
hold on
title('Convergence Behavior of FISTA and L-BFGS');
xlabel('Iteration Number') ;
ylabel('');
plot(1:length(out1.trace),(out1.trace),'LineWidth',1.2,'Color','r');
plot(1:length(out2.trace),(out2.trace),'LineWidth',1.2,'Color','blue');
legend({'FISTA','L-BFGS'},'Location','northeast')
hold off

figure;
hold on
title('PSNR of FISTA and L-BFGS');
xlabel('Iteration Number') ;
ylabel('PSNR');
plot(1:length(out1.psnr),(out1.psnr),'LineWidth',1.2,'Color','r');
plot(1:length(out2.psnr),(out2.psnr),'LineWidth',1.2,'Color','blue');
legend({'FISTA','L-BFGS'},'Location','southeast');
hold off