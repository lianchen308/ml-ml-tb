clear; clc;
fprintf('Loading data...\n');

learners = {};
weights = [];
lrn_auc = -1;
load binarySvmAdaboostModelData.mat; % learners weights lrn_auc
load('../data/binaryData.mat'); % X_train y_train X_val y_val X_test y_test

cur_learners = learners;
cur_weights = weights;


%training config
learn_obj.train = 'mabSvmTrain';
learn_obj.predict ='mabSvmPredict';
learn_obj.max_fail = 4;

X = [X_train X_val];
y = [y_train y_val];

n = 10;
n_train = 10000;
for i=1:n
    % Training
    tic;
    fprintf('Model adaboost %d of %d train...\n', i, n);
    
    rand_idx = randperm(length(y), n_train);
    [cur_learners, cur_weights, ~] = maboost(learn_obj, X(:, rand_idx), y(:, rand_idx));
    fprintf('Evaluating complete modest adaboost model...\n');
    [cur_lrn_auc, cur_lrn_acc] = mabeval(cur_learners, cur_weights, X, y, X_test, y_test);
    
    % Saving
    if (cur_lrn_auc > lrn_auc)
        fprintf('Saving adaboost model...\n');
        learners = cur_learners;
        lrn_auc = cur_lrn_auc;
        weights = cur_weights;
        save binarySvmAdaboostModelData.mat learners weights lrn_auc;
        fprintf('Adaboost model saved...\n');
    else
        fprintf('Best auc is: %1.4f\n', lrn_auc);
    end
    toc;
    fprintf('\n');
end