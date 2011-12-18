clear; clc;
fprintf('Loading data...\n');

nn_model.score = -1;
load('../../data/parsedData.mat');

if (exist('nnModelData.mat', 'file'))
    load nnModelData.mat;
end

% adjusting data format
data.x_train1 = data.x_train1';
data.y_train1 = data.y_train1';

data.y_test1 = data.y_test1';
data.x_test1 = data.x_test1';

% merging train data
data.x_train = [data.x_train1 data.x_test1];
data.y_train = [data.y_train1 data.y_test1];

n = 100;
model_hist = {};
weights = deftrainweight(data.y_train);
weights(data.y_train == 1) = weights(data.y_train == 1)*0.95;
score_fcn = 'giniscore';
for i=1:n
    % Training
    tic;
    fprintf('Model nn %d of %d train...\n', i, n); 
    
    nn_config = newff(minmax(data.x_train), minmax(data.y_train), ...
        50, {'tansig', 'tansig'});
    nn_config.trainParam.max_fail = 30;
    nn_config.trainParam.min_grad = 1e-30;
    nn_config.divideFcn = 'divideblock';
    nn_config.divideParam.trainRatio = 0.8; 
    nn_config.divideParam.valRatio = 0.15;
    nn_config.divideParam.testRatio = 0.5;
    [cur_nn_model.model] = train(nn_config, data.x_train, ...
        data.y_train, [], [], weights);
    
    [cur_nn_model.score, ~] = nnetEval(cur_nn_model.model, data.x_train1, data.y_train1, ...
        data.x_test1, data.y_test1, score_fcn);
    
    % Saving
    if (cur_nn_model.score > nn_model.score)
        fprintf('Saving nn model...\n');
        nn_model = cur_nn_model;
        save nnModelData.mat nn_model;
        fprintf('Nn model saved...\n');
    else
        fprintf('Best %s is: %1.4f\n', score_fcn, nn_model.score);
    end
    toc;
    fprintf('\n');
    model_hist{i} = cur_nn_model;    %#ok<SAGROW>
end