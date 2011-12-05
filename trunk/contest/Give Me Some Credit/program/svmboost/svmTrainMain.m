clear; clc;
fprintf('Loading data...\n');

learners = {};
weights = [];
lrn_auc = -1;
load('../nnetboost/binaryAdaboostModelData.mat') % learners weights lrn_auc
load('../data/binaryData.mat'); % X_train y_train X_val y_val X_test y_test

cur_learners = learners;
cur_weights = weights;


%training config
learn_obj.train = 'mabSvmTrain';
learn_obj.predict ='mabSvmPredict';
learn_obj.max_fail = 4;

X = [X_train X_val];
y = [y_train y_val];

retrain = 1;
n = 20;
for i=1:n
    % Training
    tic;
    fprintf('Model adaboost %d of %d train...\n', i, n);
    
    if (retrain)
        mabeval(cur_learners, cur_weights, X, y, X_test, y_test);
        [cur_learners, cur_weights] = mablrnagg(learners, weights);
        [cur_learners, cur_weights, ~] = maboost(learn_obj, X, y, cur_weights, {cur_learners});
        %retrain = 0;
    else
        [cur_learners, cur_weights, ~] = maboost(learn_obj, X(:, rand_idx), y(:, rand_idx));
    end
    
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