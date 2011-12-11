clear; clc;

svm_model_auc = 0;
fprintf('Loading data...\n');
load ../data/binaryData.mat; 
load binarySvmModelData.mat;


c_values = logspace(1, 8, 10); gamma_values = logspace(-6, 1, 10); % -> search!

n_find_params = 3000;
n_actual_train= 30000;
cross_validation = 0;

% Training
tic;
[curr_svm_model, curr_svm_model_auc, curr_c, curr_gamma] = svmgridsearch(X_train1, y_train1, X_val, y_val, X_test, y_test, ...
    c_values, gamma_values, cross_validation, n_find_params, n_actual_train);

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