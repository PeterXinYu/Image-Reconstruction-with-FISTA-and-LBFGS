function [damaged_img_vec,undamaged_img_vec,mask_vec,omg,b_vec] = get_img_para(undamaged_img_path,mask_path)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
undamaged_img = im2double(imread(undamaged_img_path));
% mask = im2double(imread('640_640_random90.png'));
mask = im2double(imread(mask_path)); %0.05
undamaged_img = undamaged_img(:,:,1);
damaged_img = mask .* undamaged_img;

mask_vec = mask(:); %ind
undamaged_img_vec = undamaged_img(:); %原图的矩阵转成vector
damaged_img_vec = damaged_img(:); %损图的矩阵转成vector

omg = find(mask_vec == 1); %big omg

b_vec = undamaged_img_vec(omg);
end

