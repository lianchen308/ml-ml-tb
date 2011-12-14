%clear; 
clc;

fprintf('Loading data...\n');

learners = {};
weights = [];
lrn_auc = 0;

load('../data/binaryData.mat');
X_train = [X_train1; X_val];
y_train = [y_train1; y_val];

X_test = [X_test; X_train2];
y_test = [y_test; y_train2];
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
learners_hist = {};
weight_hist = {};
retrain = 0;
for i=1:n
    % Training
    tic;
    fprintf('Model adaboost %d of %d train...\n', i, n);
    
    if (retrain)
        mabeval(cur_learners, cur_weights, X_train, y_train, X_test, y_test);
        [cur_learners, cur_weights] = mablrnagg(learners, weights);
        [cur_learners, cur_weights, ~] = maboost(learn_obj, X_train, y, cur_weights, {cur_learners});
        retrain = 0;
    else
        [cur_learners, cur_weights, ~] = maboost(learn_obj, X_train, y_train);
    end
    fprintf('Evaluating complete modest adaboost model...\n');
    [cur_lrn_auc, ~] = mabeval(cur_learners, cur_weights, X_train, y_train, X_test, y_test);
    
    learners_hist{i} = cur_learners; %#ok<SAGROW>
	weight_hist{i} = cur_weights; %#ok<SAGROW>
    
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