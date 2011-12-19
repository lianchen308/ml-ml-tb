clear; clc;

svm_ru_model_auc = -1;
fprintf('Loading resampled data...\n');
load ../data/binaryData.mat;
load binaryResampledData;
if (exist('binarySvmRUModelData.mat', 'file'))
    load binarySvmRUModelData.mat;
end

[rnd_sample] = shuffle([X_train1_resampled y_train1_resampled]);
X_train1_resampled = rnd_sample(:, 1:end-1);
y_train1_resampled = rnd_sample(:, end);

c_values = logspace(3, 8, 10); gamma_values = logspace(-6, -2, 10); % -> search!
n_find_params = 5000;
n_actual_train= length(y_train1_resampled);
options.weights = 1;

% Training
tic;
[curr_svm_ru_model, curr_svm_ru_model_auc, curr_ru_c, curr_ru_gamma] = svmgridsearch(X_train1_resampled, y_train1_resampled, X_val, y_val, X_test, y_test, ...
    c_values, gamma_values, options, n_find_params, n_actual_train);

[~, ~, ~, y_train2_auc] = svmpredictw(curr_svm_ru_model, X_train2, y_train2);
fprintf('AUC: train2 auc = %1.4f\n', y_train2_auc);

curr_svm_ru_model_auc = min(y_train2_auc, curr_svm_ru_model_auc);

% Saving
if (curr_svm_ru_model_auc > svm_ru_model_auc)
    fprintf('Saving svmru model...\n');
    svm_ru_model = curr_svm_ru_model;
    svm_ru_model_auc = curr_svm_ru_model_auc;
    svm_ru_c = curr_ru_c; 
    svm_ru_gamma = curr_ru_gamma;
    save('binarySvmRUModelData.mat', 'svm_ru_model', 'svm_ru_model_auc', 'svm_ru_c', 'svm_ru_gamma');
    fprintf('Svm modelru saved...\n');
else
    fprintf('Best auc is: %1.4f\n', svm_ru_model_auc);
end
toc;
fprintf('\n');