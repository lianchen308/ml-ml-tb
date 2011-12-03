clear; clc;
fprintf('Loading data...\n');

load('../data/binaryData.mat'); % X_train y_train X_val y_val X_test y_test
X_train = X_train'; y_train = y_train'; X_val = X_val'; y_val = y_val'; X_test = X_test'; y_test = y_test'; 
load binarySvmModelData.mat; % svm_model svm_model_auc

%c_values = 483293.0238571752; gamma_values = 0.0000615848; % -> acc:93.6322, auc:0.8442
%c_values = 1438449.8882876630; gamma_values = 0.000089; % -> acc:93.6043, auc:0.8441
%c_values = 5994842.50318941; gamma_values = 0.0000215443;
c_values = logspace(6, 8, 10); gamma_values = logspace(-6, -4, 10); % -> search!
n_find_params = 6000;
n_actual_train= 20000;
cross_validation = 0;
% Training
tic;
[curr_svm_model, curr_svm_model_auc] = svmgridsearch(X_train, y_train, X_val, y_val, X_test, y_test, ...
    c_values, gamma_values, cross_validation, n_find_params, n_actual_train);

% Saving
if (curr_svm_model_auc > svm_model_auc)
    fprintf('Saving svm model...\n');
    svm_model = curr_svm_model;
    svm_model_auc = curr_svm_model_auc;
    save binarySvmModelData.mat svm_model svm_model_auc;
    fprintf('Svm model saved...\n');
else
    fprintf('Best auc is: %1.4f\n', svm_model_auc);
end
toc;
fprintf('\n');