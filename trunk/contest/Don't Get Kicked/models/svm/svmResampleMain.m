clear; clc;

svm_model_auc = -1;
fprintf('Loading data...\n');

load ../../data/parsedData.mat;

tic;
fprintf('Resampling data...\n');
c = 2.154434690031882e+02; gamma = 2.782559402207126e-04;
n_batch = 45000;
score_fcn = 'giniscore';
[resampled_data.x_train1_resampled, resampled_data.y_train1_resampled] = ...
    gsvmru(data.x_train1, data.y_train1, data.x_test1, data.y_test1, ...
        c, gamma, score_fcn, n_batch);
fprintf('Data resampled...\n');
save resampledData.mat resampled_data;
toc;
fprintf('\n');

