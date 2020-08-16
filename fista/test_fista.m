[damaged_img_vec,undamaged_img_vec,mask_vec,omg,b_vec] = get_img_para('./test_images/640_640_lion.png','./test_masks/640_640_mesh.png');


opts.mu = 0.01;
opts.tol = 1e-3;

tic
[x,out]    = fista(f,phi,x0,opts,undamaged_img_vec);
psnr_value  = psnr(undamaged_img_vec, x);
t           = toc;
fprintf(1,'--- fista: [ITER; OBJ]: [%4i; %1.2e]; TIME: %2.2f sec; PSNR: %2.2f \n',length(out.trace),out.trace(end),t,psnr_value);