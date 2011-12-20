%clear; 
clc;

fprintf('Loading data...\n');

mab_nn_model.score = -1;
load('../../data/parsedData.mat');

if (exist('mabNnModelData.mat', 'file'))
    load mabNnModelData.mat;
end
data.x_train = [data.x_train1; data.x_test1; data.x_train2; data.x_test2];
data.y_train = [data.y_train1; data.y_test1; data.y_train2; data.y_test2];

curr_mab_nn_model = mab_nn_model;

%training config
learn_obj.train = 'mabnnTrain';
learn_obj.predict ='nnetBoostSim';
learn_obj.score_fcn = 'giniscore';
learn_obj.max_fail = 4;
learn_obj.X_val = [data.x_train2; data.x_test2];
learn_obj.y_val = [data.y_train2; data.y_test2];


n = 1000;
model_hist = {};
retrain = 0;
for i=1:n
    % Training
    tic;
    fprintf('Mab nn %d of %d train...\n', i, n);
    
    if (retrain)
        mabeval(cur_learners, cur_weights, X_train, y_train, X_test, y_test);
        [curr_mab_nn_model.learners, curr_mab_nn_model.weights, ~] = ...
            maboost(learn_obj, data.x_train, data.y_train, ...
            mab_nn_model.weights, mab_nn_model.learners);
        retrain = 0;
    else
        [curr_mab_nn_model.learners, curr_mab_nn_model.weights, ~] = ...
            maboost(learn_obj, data.x_train, data.y_train);
    end
    fprintf('Evaluating complete Mab nn model...\n');
    [~, ~, ~, curr_mab_nn_model.score] = mabeval(curr_mab_nn_model.learners, curr_mab_nn_model.weights, ...
        data.x_train1, data.y_train1, learn_obj.X_val, learn_obj.y_val, learn_obj.score_fcn);
    
    
    % Saving
    if (curr_mab_nn_model.score > mab_nn_model.score)
        fprintf('Saving mab nn model...\n');
        mab_nn_model = curr_mab_nn_model;
        save mabNnModelData.mat mab_nn_model;
        fprintf('Mab nn model saved...\n');
    else
        fprintf('Best %s is: %1.4f\n', learn_obj.score_fcn, mab_nn_model.score);
    end
    
    model_hist{i} = curr_mab_nn_model; %#ok<SAGROW>
    
    toc;
    fprintf('\n');
end