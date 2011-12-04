%clear; 
clc;

fprintf('Loading data...\n');

learners = {};
weights = [];
lrn_auc = -1;
load binaryAdaboostModelData.mat; % learners weights lrn_auc
load('../data/binaryData.mat'); % X_train y_train X_val y_val X_test y_test
cur_learners = learners;
cur_weights = weights;

%training config
learn_obj.train = 'mabNnetTrain';
learn_obj.predict ='sim';
learn_obj.max_fail = 7;

n = 1000;
learners_hist = cell(0,0);
weight_hist = cell(0,0);
retrain = 0;
X = [X_train X_val];
y = [y_train y_val];
for i=1:n
    % Training
    tic;
    fprintf('Model adaboost %d of %d train...\n', i, n);
    
    if (retrain)
        mabeval(cur_learners, cur_weights, X, y, X_test, y_test);
        [cur_learners, cur_weights] = mablrnagg(learners, weights);
        [cur_learners, cur_weights, ~] = maboost(learn_obj, X, y, cur_weights, {cur_learners});
        retrain = 0;
    else
        [cur_learners, cur_weights, ~] = maboost(learn_obj, X, y);
    end
    fprintf('Evaluating complete modest adaboost model...\n');
    [cur_lrn_auc, cur_lrn_acc] = mabeval(cur_learners, cur_weights, X, y, X_test, y_test);
    learners_hist{i} = cur_learners;
	weight_hist{i} = cur_weights;
    
    % Saving
    if (cur_lrn_auc > lrn_auc)
        fprintf('Saving adaboost model...\n');
        learners = cur_learners;
        lrn_auc = cur_lrn_auc;
        weights = cur_weights;
        save binaryAdaboostModelData.mat learners weights lrn_auc;
        fprintf('Adaboost model saved...\n');
    else
        fprintf('Best auc is: %1.4f\n', lrn_auc);
    end
    toc;
    fprintf('\n');
end