clear; clc;
fprintf('Loading data...\n');

gaussian_model_auc = -1;
load ../data/binaryData.mat;
if (exist('binaryGaussianModelData.mat', 'file'))
    load binaryGaussianModelData.mat; 
end


% Training
tic;
fprintf('Model Gaussian train...\n');
[cur_gaussian_model] = gaussiantrain([X_train1; X_val], [y_train1; y_val], 'aucscore');
[curr_gaussian_model_auc, ~] = gaussianeval(cur_gaussian_model, X_test, y_test);

% Saving
if (curr_gaussian_model_auc > gaussian_model_auc)
    fprintf('Saving gaussian model...\n');
    gaussian_model = cur_gaussian_model;
    gaussian_model_auc = curr_gaussian_model_auc;
    save binaryGaussianModelData.mat gaussian_model gaussian_model_auc;

    fprintf('Gaussian saved...\n');
else
    fprintf('Best auc is: %1.4f\n', gaussian_model_auc);
end
toc;
fprintf('\n');