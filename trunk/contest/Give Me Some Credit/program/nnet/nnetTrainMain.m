clear; clc;
fprintf('Loading data...\n');

nn_model_auc = -1;
load('../data/binaryData.mat');
X_train = [X_train1' X_val']; X_train1 = X_train1'; X_val = X_val'; X_test = [X_test' X_train2'];
y_train = [y_train1' y_val']; y_train1 = y_train1'; y_val = y_val'; y_test = [y_test' y_train2'];

if (exist('binaryNnetModelData.mat', 'file'))
    load binaryNnetModelData.mat;
end

n = 100;
model_hist = cell(n, 1);
auc_hist = zeros(n, 1);
weights = deftrainweight(y_train); % ones(size(y_train));
weights(y_train == 1) = weights(y_train == 1)*1.2;
    
for i=1:n
    % Training
    tic;
    fprintf('Model nnet %d of %d train...\n', i, n);
    [cur_nn_model] = nnetTrain(X_train, y_train, 18, {'tansig', 'tansig'}, weights);
    [curr_nn_model_auc, ~] = nnetEval(cur_nn_model, X_train1, y_train1, X_val, y_val, X_test, y_test);
    
    % Saving
    if (curr_nn_model_auc > nn_model_auc)
        fprintf('Saving nn model...\n');
        nn_model = cur_nn_model;
        nn_model_auc = curr_nn_model_auc;
        save binaryNnetModelData.mat nn_model nn_model_auc;
        fprintf('Nn model saved...\n');
    else
        fprintf('Best auc is: %1.4f\n', nn_model_auc);
    end
    toc;
    fprintf('\n');
    model_hist{i} = cur_nn_model;  
    auc_hist(i) = curr_nn_model_auc;  
end