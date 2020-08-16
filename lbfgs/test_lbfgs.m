

[damaged_img_vec,undamaged_img_vec,mask_vec,omg,b_vec] = get_img_para('../../test_images/640_640_lion_grey.png','../../test_masks/640_640_handwriting.png');
[f,x0,opts] = get_para_lbfgs(damaged_img_vec,undamaged_img_vec,mask_vec,omg,b_vec);

tic
[x,out]    = lbfgs(f,x0,opts,undamaged_img_vec);
psnr_value  = psnr(undamaged_img_vec, x);
t           = toc;
fprintf(1,'--- fista: [ITER; OBJ]: [%4i; %1.2e]; TIME: %2.2f sec; PSNR: %2.2f\n',length(out.trace),out.trace(end),t, psnr_value);

plot(1:length(out.trace),log10(out.trace),'LineWidth',1.2,'Color','r');





