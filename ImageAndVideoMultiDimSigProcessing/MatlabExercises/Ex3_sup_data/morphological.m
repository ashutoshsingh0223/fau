image = imread('morph_object_matlab.png');

square = strel('square', 3);

eroded = imerode(image, square);
dilated = imdilate(image, square);

opened = imerode(dilated, square);
closed = imdilate(eroded, square);

subplot(2,2,1)
imshow(dilated);
title('dilated')

subplot(2,2,2)
imshow(eroded);
title('eroded')

subplot(2,2,3)
imshow(opened);
title('opened')

subplot(2,2,4)
imshow(closed);
title('closed')