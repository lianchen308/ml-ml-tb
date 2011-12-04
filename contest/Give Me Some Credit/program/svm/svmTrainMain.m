clear; clc;

svm_model_auc = 0;
fprintf('Loading data...\n');
load('../data/binaryData.mat'); % X_train y_train X_val y_val X_test y_test
X_train = X_train'; y_train = y_train'; X_val = X_val'; y_val = y_val'; X_test = X_test'; y_test = y_test';
load('binarySvmModelData.mat'); % 'svm_model', 'svm_model_auc', 'svm_c', 'svm_gamma'

use_resample = 1;
c_values = logspace(6, 8, 10); gamma_values = logspace(-6, -4, 10); % -> search!
%c_values = 7742636.8268112773; gamma_values = 0.0000215443; % -> acc:93.6132, auc:0.8506!
n_find_params = 6000;
n_actual_train= 20000;
cross_validation = 0;

if (use_resample)
    fprintf('Loading resampling data...\n');
    load('binarySvmResampledData.mat'); %  X_train_resampled y_train_resampled
    idx = ismember(X_train, X_train_resampled, 'rows');
    X_train = X_train_resampled;
    y_train = y_train_resampled;
    [X_train, idx] = unique(X_train, 'rows');
    y_train = y_train(idx);
end

% Training
tic;
[curr_svm_model, curr_svm_model_auc, curr_c, curr_gamma] = svmgridsearch(X_train, y_train, X_val, y_val, X_test, y_test, ...
    c_values, gamma_values, cross_validation, n_find_params, n_actual_train);

if (use_resample)
    [~, ~, ~, y_val_auc] = svmpredictw(curr_svm_model, X_val, y_val);
    [~, ~, ~, y_test_auc] = svmpredictw(curr_svm_model, X_test, y_test);
    curr_svm_model_auc = min([y_val_auc y_test_auc]);
end

% Saving
if (curr_svm_model_auc > svm_model_auc)
    fprintf('Saving svm model...\n');
    svm_model = curr_svm_model;
    svm_model_auc = curr_svm_model_auc;
    svm_c = curr_c; 
    svm_gamma = curr_gamma;
    save('binarySvmModelData.mat', 'svm_model', 'svm_model_auc', 'svm_c', 'svm_gamma');
    fprintf('Svm model saved...\n');
else
    fprintf('Best auc is: %1.4f\n', svm_model_auc);
end
toc;
fprintf('\n');