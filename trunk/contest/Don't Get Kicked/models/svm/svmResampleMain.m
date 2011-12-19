clear; clc;

svm_model_auc = -1;
fprintf('Loading data...\n');
load ../data/binaryData.mat;

tic;
fprintf('Resampling data...\n');
c = 359381.36638046; gamma = 0.0000464159;
n_batch = 45000;
[X_train1_resampled, y_train1_resampled] = gsvmru(X_train1, y_train1, X_val, y_val, c, gamma, n_batch);
fprintf('Data resampled...\n');
save binaryResampledData.mat X_train1_resampled y_train1_resampled;
toc;
fprintf('\n');

