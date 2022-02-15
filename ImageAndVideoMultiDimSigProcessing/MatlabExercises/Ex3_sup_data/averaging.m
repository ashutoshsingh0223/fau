shed1 = imread("shed_1.png");

noise1 = uint16(imnoise(shed1,'gaussian',0,0.01));
noise2 = uint16(imnoise(shed1,'gaussian',0,0.01));
noise3 = uint16(imnoise(shed1,'gaussian',0,0.01));
noise4 = uint16(imnoise(shed1,'gaussian',0,0.01));
noise5 = uint16(imnoise(shed1,'gaussian',0,0.01));

out = (noise1 + noise2 + noise3 + noise4 + noise5)./ 5;

subplot(3,3,1)
imshow(shed1);
title('Original')

subplot(3,3,2)
imshow(uint8(out))
title('Averaged')

subplot(3,3,3)
imshow(uint8(noise1))
title('Noise1')

subplot(3,3,4)
imshow(uint8(noise2))
title('Noise2')

subplot(3,3,5)
imshow(uint8(noise3))
title('Noise3')

subplot(3,3,6)
imshow(uint8(noise4))
title('Noise4')

subplot(3,3,7)
imshow(uint8(noise5))
title('Noise5')

