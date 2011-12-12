clear; 
clc;

fprintf('Loading data...\n');

nnet_mix_learners = {};
nnet_mix_weights = [];
nnet_mix_lrn_auc = -1;

load ../data/binaryData.mat;
load binaryMixData.mat;
X_train1 = X_train1_mix;
X_train2 = X_train2_mix;
X_val = X_val_mix;
X_test = X_test_mix;

if (exist('binaryMixBoostModelData.mat', 'file'))
    load binaryMixBoostModelData.mat;
end

cur_learners = nnet_mix_learners;
cur_weights = nnet_mix_weights;

%training config
learn_obj.train = 'nnetTrain';
learn_obj.predict ='nnetBoostSim';
learn_obj.max_fail = 6;
learn_obj.X_val = X_test;
learn_obj.y_val = y_test;

n = 1000;
learners_hist = cell(0,0);
weight_hist = cell(0,0);
retrain = 0;
X = [X_train2; X_test];
y = [y_train2; y_test];
for i=1:n
    % Training
    tic;
    fprintf('Mixed model adaboost %d of %d train...\n', i, n);
    
    if (retrain)
        [cur_learners, cur_weights, ~] = maboost(learn_obj, X, y, nnet_mix_weights, nnet_mix_learners);
        retrain = 0;
    else
        [cur_learners, cur_weights, ~] = maboost(learn_obj, X, y);
    end
    fprintf('Evaluating complete modest adaboost model...\n');
    fprintf('Train1 and validation\n');
    mabeval(cur_learners, cur_weights, X_train1, y_train1,  X_val, y_val);
    fprintf('Train2 and test\n');
    [cur_lrn_auc, cur_lrn_acc] = mabeval(cur_learners, cur_weights, X_train2, y_train2, X_test, y_test);
    learners_hist{i} = cur_learners;
	weight_hist{i} = cur_weights;
    
    % Saving
    if (cur_lrn_auc > nnet_mix_lrn_auc)
        fprintf('Saving adaboost model...\n');
        nnet_mix_learners = cur_learners;
        nnet_mix_lrn_auc = cur_lrn_auc;
        nnet_mix_weights = cur_weights;
        save binaryMixBoostModelData.mat nnet_mix_learners nnet_mix_weights nnet_mix_lrn_auc;
        fprintf('Adaboost model saved...\n');
    else
        fprintf('Best auc is: %1.4f\n', nnet_mix_lrn_auc);
    end
    toc;
    fprintf('\n');
end