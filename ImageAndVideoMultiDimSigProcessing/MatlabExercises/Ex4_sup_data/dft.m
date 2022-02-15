image = imread("london_panorama.jpg");
grey = rgb2gray(image);

filter = fspecial('gaussian', 199, 20);

I = size(filter,1);
[M, N] = size(grey);

filt_pad = padarray(filter, [M-I, N-I], "post");

filt_dft = fft2(filt_pad);
grey_dft = fft2(grey);

result = grey_dft .* filt_dft;

result = ifft2(result);
subplot(2,2,1)
imshow(uint8(result));

shift_filt_pad = circshift(filt_pad, [-(I-1)/2 -(I-1)/2]);
shift_filt_pad_dft = fft2(shift_filt_pad);


result2 = grey_dft .* shift_filt_pad_dft;

result2 = ifft2(result2);
subplot(2,2,2)
imshow(uint8(result2));


padx = N + I - 1;
pady = M + I - 1;


grey_padded = padarray(grey, [(pady-M)/2 (padx-N)/2]);
grey_padded_dft = fft2(grey_padded);

filt_padded = padarray(filter, [(pady - I) (padx - I)], "post");
filt_padded_dft = fft2(filt_padded);


result3 = grey_padded_dft .* filt_padded_dft;
result3 = ifft2(result3);
result3 = result3(pady-M+1: end, padx-N+1: end);

subplot(2,2,3)
imshow(uint8(result3));






grey_padded_symm = padarray(grey, [(pady-M)/2 (padx-N)/2], "symmetric");
grey_padded_symm_dft = fft2(grey_padded_symm);

result4 = grey_padded_symm_dft .* filt_padded_dft;
result4 = ifft2(result4);
% result4 = result4(pady-M+1: end, padx-N+1: end);

subplot(2,2,4)
imshow(uint8(result4));








