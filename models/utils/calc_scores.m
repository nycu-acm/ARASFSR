function scores = calc_scores(input_dir,shave_width,verbose,resize)

addpath(genpath(fullfile(pwd,'utils')));

%% Loading model
load modelparameters.mat
blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;

%% Reading file list
file_list = {};
exts = {'.jpg','.jpeg','.png','.tif','.bmp'};
for i = 1 : length(exts)
    temp_files = dir(fullfile(input_dir,['*' exts{i}]));
    file_list = [file_list; {temp_files(:).name}'];
end
%% file_list = dir([input_dir,'/*.jpg']);
im_num = length(file_list);

%% Calculating scores
scores = struct([]);

for ii=1:im_num
    if verbose
        fprintf(['\nCalculating scores for image ',num2str(ii),' / ',num2str(im_num)]);
    end
    
    % Reading and converting images
    input_image_path = fullfile(input_dir,file_list{ii});
    image = imread(input_image_path);
    if resize
        [h, w] = size(image, [1, 2]);
        % Resize the shortest side to 512
        if h > w
            image = imresize(image, [NaN 512]);
        else
            image = imresize(image, [512 NaN]);
        end
    end
    input_image = convert_shave_image(image,shave_width);
    % GD_image_path = fullfile(GT_dir,file_list(ii).name);
    % GD_image = convert_shave_image(imread(GD_image_path),shave_width);
    
    % Calculating scores
    scores(ii).name = file_list{ii};
    % scores(ii).MSE = immse(input_image,GD_image);
    scores(ii).Ma = quality_predict(input_image);
    scores(ii).NIQE = computequality(input_image,blocksizerow,blocksizecol,...
        blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam);
    scores(ii).PI = (scores(ii).NIQE + (10 - scores(ii).Ma)) / 2;
end

end
