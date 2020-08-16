function psnr_value = psnr(u,x)
    mn = size(u,1);
    y = idct(x);
    psnr_value = 10 * log10((mn)/(norm(y - u)^2));