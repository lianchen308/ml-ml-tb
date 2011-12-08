clear; clc;
fprintf('Loading data...\n');

use_pca = 0;
if (use_pca)
    load('../data/binaryPCAData.mat'); % Z_train y_train Z_val y_val Z_test y_test
    X_train = Z_train'; X_val = Z_val'; X_test = Z_test';
    load binaryNnetModelPCAData.mat; % nn_model nn_model_auc
else
    load('../data/binaryData.mat'); % X_train y_train X_val y_val X_test y_test
    X_train = X_train'; X_val = X_val'; X_test = X_test';
    load binaryNnetModelData.mat; % nn_model nn_model_auc
end
y_train = y_train'; y_val = y_val'; y_test = y_test';

n = 5;
model_hist = cell(n, 1);
auc_hist = zeros(n, 1);
for i=1:n
    % Training
    tic;
    fprintf('Model nnet %d of %d train...\n', i, n);
    [cur_nn_model] = nnetTrain([X_train X_val], [y_train y_val], 15, {'tansig', 'tansig'});
    [curr_nn_model_auc, ~] = nnetEval(cur_nn_model, X_train, y_train, X_val, y_val, X_test, y_test);
    
    % Saving
    if (curr_nn_model_auc > nn_model_auc)
        fprintf('Saving nn model...\n');
        nn_model = cur_nn_model;
        nn_model_auc = curr_nn_model_auc;
        if (use_pca)
            save binaryNnetModelPCAData.mat nn_model nn_model_auc;
        else
            save binaryNnetModelData.mat nn_model nn_model_auc;
        end
        
        fprintf('Nn model saved...\n');
    else
        fprintf('Best auc is: %1.4f\n', nn_model_auc);
    end
    toc;
    fprintf('\n');
    model_hist{i} = cur_nn_model;  
    auc_hist(i) = curr_nn_model_auc;  
end