img = imread('f:\\1.jpg');
gray_256 = rgb2gray(img);
gray_16 = (gray_256 - rem(gray_256, 16)) / 16;
gray_max = max(max(gray_16));
gray_min = min(min(gray_16));
numlevels = gray_max - gray_min + 1;
% gray_16 = [[1, 1, 5, 6, 8];
%         [2, 3, 5, 7, 1];
%         [4, 5, 7, 1, 2];
%         [8, 5, 1, 2, 5]];
offsets = [
    [0, 1];
    [1, 1];
    [1, 0];
    [-1, 1]
    ];
[m, n] = size(offsets);
for i = 1: m
    glcm = graycomatrix(gray_16, 'Numlevels', numlevels, 'g', [], 'offset', [offsets(i, 1), offsets(i, 2)]);
    prop = graycoprops(glcm, 'all');
    disp(prop);
end
