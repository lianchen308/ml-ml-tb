clear; clc;

tic;

if (~exist('rawData.mat', 'file'))
    runBuildRawDataMain;
end
load rawData.mat;
data = raw_data;

% random seed fixed to compare results
rng(465354);

options.k = [10 15 20];

fprintf('x_train analisys: \n\n');
imputeanalisys(raw_data.x_train, raw_data.is_discrete, options);

fprintf('x_submit analisys: \n\n');
imputeanalisys(raw_data.x_submit, raw_data.is_discrete, options);

toc;