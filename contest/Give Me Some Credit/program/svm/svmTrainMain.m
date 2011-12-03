clear; clc;
fprintf('Loading data...\n');

load('../data/binaryData.mat'); % X_train y_train X_val y_val X_test y_test
X_train = X_train'; y_train = y_train'; X_val = X_val'; y_val = y_val'; X_test = X_test'; y_test = y_test'; 
%load binarySvmModelData.mat; % svm_model svm_model_auc
svm_model_auc = 0;

c_values = 1000000000; gamma_values = 0.0000215443; % -> 
%c_values = logspace(1, 9, 10); gamma_values = logspace(-8, 1, 10); % -> search!

n = 1;
model_hist = cell(n, 1);
auc_hist = zeros(n, 1);
for i=1:n
    % Training
    tic;
    fprintf('Model svm %d of %d train...\n', i, n);
    [curr_svm_model, curr_svm_model_auc] = svmAutoTrain(X_train, y_train, X_val, y_val, X_test, y_test, c_values, gamma_values);
    
    % Saving
    if (curr_svm_model_auc > svm_model_auc)
        fprintf('Saving svm model...\n');
        svm_model = cur_svm_model;
        svm_model_auc = curr_svm_model_auc;
        save binarySvmModelData.mat svm_model svm_model_auc;
        fprintf('Svm model saved...\n');
    else
        fprintf('Best auc is: %1.4f\n', svm_model_auc);
    end
    toc;
    fprintf('\n');
    model_hist{i} = cur_svm_model;  
    auc_hist(i) = curr_svm_model_auc;  
end