clear; clc;
fprintf('Loading data...\n');

load('../data/binaryData.mat'); % X_train y_train X_val y_val X_test y_test
load binaryNnetModelData.mat; % nn_model nn_model_auc

n = 200;
model_hist = cell(n, 1);
auc_hist = zeros(n, 1);
for i=1:n
    % Training
    tic;
    fprintf('Model nnet %d of %d train...\n', i, n);
    [cur_nn_model] = nnetTrain([X_train X_val], [y_train y_val], [10 10], {'tansig', 'tansig', 'tansig'});
    [curr_nn_model_auc, ~] = nnetEval(cur_nn_model, X_train, y_train, X_val, y_val, X_test, y_test);
    
    % Saving
    if (curr_nn_model_auc > nn_model_auc)
        fprintf('Saving nn model...\n');
        nn_model = cur_nn_model;
        nn_model_auc = curr_nn_model_auc;
        %save binaryNnetModelData.mat nn_model nn_model_auc;
        fprintf('Nn model saved...\n');
    else
        fprintf('Best auc is: %1.4f\n', nn_model_auc);
    end
    toc;
    fprintf('\n');
    model_hist{i} = cur_nn_model;  
    auc_hist(i) = curr_nn_model_auc;  
end