function Prepare_TrainData_HR_LR()
clear; close all; clc

% SET scale factor here
degradation = 'BI'; % BI, BD, DN
if strcmp(degradation, 'BI')
    scale = 4; % 3, 4, 8;
else
    scale = 3;
end

% SET data dir
sourcedir = '/HR_folder';
savedir = '/Augmentation/train/BI';
saveHRpath = fullfile(savedir, 'Train_HR_aug', ['x' num2str(scale)]);
%saveLRpath = fullfile(savedir, 'Train_LR_aug', ['x' num2str(scale)]);

% downsizes = [0.9, 0.8, 0.7, 0.6, 0.5];

if ~exist(saveHRpath, 'dir')
    mkdir(saveHRpath);
end
%if ~exist(saveLRpath, 'dir')
%    mkdir(saveLRpath);
%end

filepaths = [dir(fullfile(sourcedir, '*.png'));dir(fullfile(sourcedir, '*.bmp'))];
% kernelsize = 7;
% prepare data with augmentation
parfor i = 1 : length(filepaths)
    filename = filepaths(i).name;
    fprintf('No.%d -- Processing %s\n', i, filename);
    [add, im_name, type] = fileparts(filepaths(i).name);
    image = imread(fullfile(sourcedir, filename));
    
    for angle = 0 : 1 : 3
%     for angle = 0 : 2 : 2
%         for downidx = 0 : 1 : length(downsizes)
%             image_HR = image;
%             if downidx > 0
%                 image_HR = imresize(image_HR, downsizes(downidx), 'bicubic');
%             end
            image_HR = image;
            image_HR = rot90(image_HR, angle);
%            image_HR = modcrop(image_HR, scale);
%            image_LR = imresize(image_HR, 1/scale, 'bicubic');
%             image_LR = imresize_BD(image_HR, 3, 'Gaussian', 1.6);
%             image_LR = imresize_DN(image_HR, scale, 30);
%             saveHRfile =  [im_name '_rot' num2str(angle*90) '_ds' num2str(downidx) '.png'];
%             saveLRfile = [im_name '_rot' num2str(angle*90) '_ds' num2str(downidx) '.png'];
            saveHRfile =  [im_name '_rot' num2str(angle*90) '.png'];
%            saveLRfile = [im_name '_rot' num2str(angle*90) '.png'];

            imwrite(image_HR, fullfile(saveHRpath, saveHRfile));    
%            imwrite(image_LR, fullfile(saveLRpath, saveLRfile));               
        end          
    end
end

% end

function imgs = modcrop(imgs, modulo)
if size(imgs,3)==1
    sz = size(imgs);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2));
else
    tmpsz = size(imgs);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2),:);
end
end


function [LR] = imresize_BD(im, scale, type, sigma)

if nargin ==3 && strcmp(type,'Gaussian')
    sigma = 1.6;
end

if strcmp(type,'Gaussian') && fix(scale) == scale
    if mod(scale,2)==1
        kernelsize = ceil(sigma*3)*2+1;
        if scale==3 && sigma == 1.6
            kernelsize = 7;
        end
        kernel  = fspecial('gaussian',kernelsize,sigma);
        blur_HR = imfilter(im,kernel,'replicate');
        
        if isa(blur_HR, 'gpuArray')
            LR = blur_HR(scale-1:scale:end-1,scale-1:scale:end-1,:);
        else
            LR      = imresize(blur_HR, 1/scale, 'nearest');
        end
        
        
        % LR      = im2uint8(LR);
    elseif mod(scale,2)==0
        kernelsize = ceil(sigma*3)*2+2;
        kernel     = fspecial('gaussian',kernelsize,sigma);
        blur_HR    = imfilter(im, kernel,'replicate');
        LR= blur_HR(scale/2:scale:end-scale/2,scale/2:scale:end-scale/2,:);
        % LR         = im2uint8(LR);
    end
else
    LR = imresize(im, 1/scale, type);
end
end

function ImLR = imresize_DN(ImHR, scale, sigma)
% ImLR and ImHR are uint8 data
% downsample by Bicubic
ImDown = imresize(ImHR, 1/scale, 'bicubic'); % 0-255
ImDown = single(ImDown); % 0-255
ImDownNoise = ImDown + single(sigma*randn(size(ImDown))); % 0-255
ImLR = uint8(ImDownNoise); % 0-255
end
