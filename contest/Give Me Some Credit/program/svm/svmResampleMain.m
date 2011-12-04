clear; clc;

load('../data/binaryData.mat'); % X_train y_train X_val y_val X_test y_test
c = 7742636.8268112773;
gamma = 0.0000215443;
n_batch = 20000;

fprintf('Resampling data...\n');
tic;
[X_train_resampled, y_train_resampled] = gsvmru(X_train', y_train', X_val', y_val', ...
        c, gamma, n_batch);
n_resampled = length(y_train_resampled);
n_resampled_ratio = length(y_train_resampled(y_train_resampled == 1))/n_resampled;
fprintf('Resampled data length: %d. Positive/all ratio: %1.4f.\n', n_resampled, n_resampled_ratio);
toc;

save binarySvmResampledData.mat X_train_resampled y_train_resampled;