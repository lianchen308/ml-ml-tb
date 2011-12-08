%clear; 
clc;

fprintf('Loading data...\n');

learners = {};
weights = [];
lrn_auc = -1;

use_pca = 0;
if (use_pca)
    load('../data/binaryPCAData.mat'); % Z_train y_train Z_val y_val Z_test y_test
    X_train = Z_train'; X_val = Z_val'; X_test = Z_test';
    load binaryAdaboostModelPCAData.mat; % learners weights lrn_auc
else
    load('../data/binaryData.mat'); % X_train y_train X_val y_val X_test y_test
    X_train = X_train'; X_val = X_val'; X_test = X_test';
    load binaryAdaboostModelData.mat; % learners weights lrn_auc
end
y_train = y_train'; y_val = y_val'; y_test = y_test';

cur_learners = learners;
cur_weights = weights;

%training config
learn_obj.train = 'mabNnetTrain';
learn_obj.predict ='sim';
learn_obj.max_fail = 4;

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
        if (use_pca)
            save binaryAdaboostModelPCAData.mat learners weights lrn_auc;
        else
            save binaryAdaboostModelData.mat learners weights lrn_auc;
        end
        fprintf('Adaboost model saved...\n');
    else
        fprintf('Best auc is: %1.4f\n', lrn_auc);
    end
    toc;
    fprintf('\n');
end