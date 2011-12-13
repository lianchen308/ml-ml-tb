%clear; 
clc;

fprintf('Loading data...\n');

learners = {};
weights = [];
lrn_auc = 0;

load('../data/binaryData.mat');
if (exist('binaryAdaboostModelData.mat', 'file'))
    load binaryAdaboostModelData.mat; 
end

cur_learners = learners;
cur_weights = weights;

%training config
learn_obj.train = 'nnetTrain';
learn_obj.predict ='nnetBoostSim';
learn_obj.max_fail = 4;
learn_obj.X_val = X_val;
learn_obj.y_val = y_val;

n = 1000;
learners_hist = cell(0,0);
weight_hist = cell(0,0);
retrain = 0;
X = [X_train1; X_val];
y = [y_train1; y_val];
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
    fprintf('Train1 and validation: \n');
    [~, ~] = mabeval(cur_learners, cur_weights, X_train1, y_train1, X_val, y_val);
    
    [y_train2_acc, y_train2_auc] = mabscore(cur_learners, cur_weights, X_train2, y_train2);
    fprintf('Train2 result: acc = %1.4f, auc = %1.4f\n', y_train2_acc, y_train2_auc);
    [y_test_acc, y_test_auc] = mabscore(cur_learners, cur_weights, X_test, y_test);
    fprintf('Test result: acc = %1.4f, auc = %1.4f\n', y_test_acc, y_test_auc);
    
    cur_lrn_auc = min(y_train2_auc, y_test_auc);
    
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