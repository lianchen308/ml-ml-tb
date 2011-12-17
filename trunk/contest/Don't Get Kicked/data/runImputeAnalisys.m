clear; clc;

tic;

if (~exist('rawData.mat', 'file'))
    runBuildRawDataMain;
end
load rawData.mat;
data = raw_data;

fprintf('x_train analisys: \n\n');
imputeanalisys(raw_data.x_train, raw_data.is_discrete);

fprintf('x_submit analisys: \n\n');
imputeanalisys(raw_data.x_submit, raw_data.is_discrete);

toc;