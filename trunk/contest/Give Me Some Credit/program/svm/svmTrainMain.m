clear; clc;

svm_model_auc = -1;
fprintf('Loading data...\n');
load ../data/binaryData.mat;
if (exist('binarySvmModelData.mat', 'file'))
    load binarySvmModelData.mat;
end


% c_values = 2154.43469003; gamma_values = 0.0002154435; % -> AUC: train=0.8584, val=0.8568, test=0.8617
% c_values = 1000000.00000000; gamma_values = 0.0000100000; % -> AUC: train=0.8574, val=0.8561, test=0.8614
% c_values = 359381.36638046; gamma_values = 0.0000464159; % -> AUC: train=0.8588, val=0.8564, test=0.8621
c_values = 278255.9402; gamma_values = 1.6681e-005;
%c_values = logspace(1, 8, 10); gamma_values = logspace(-7, -1, 10); % -> search!
n_find_params = 3000;
n_actual_train= 45000;
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