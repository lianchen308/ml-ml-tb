clear; clc;
fprintf('Loading data...\n');

pca_nn_model.score = -1;
load('../../data/pcaParsedData.mat');
data = pcadata;

if (exist('pcaNnModelData.mat', 'file'))
    load pcaNnModelData.mat;
end

% adjusting data format
data.x_train1 = data.x_train1';
data.y_train1 = data.y_train1';

data.y_test1 = data.y_test1';
data.x_test1 = data.x_test1';

% merging train data
data.x_train = [data.x_train1 data.x_test1];
data.y_train = [data.y_train1 data.y_test1];

n = 500;
model_hist = {};

n_w = 15;
weights = linspace(1, negposratio(data.y_train)*1.5, n_w);

score_fcn = 'giniscore';

for i=1:n
    % Training
    tic;
    
    nn_config = newff(minmax(data.x_train), minmax(data.y_train), ...
        60, {'tansig', 'tansig', 'tansig', 'tansig'});
    nn_config.trainParam.max_fail = 10;
    nn_config.trainParam.min_grad = 1e-30;
    nn_config.divideFcn = 'divideblock';
    nn_config.divideParam.trainRatio = 0.8; 
    nn_config.divideParam.valRatio   = 0.2;
    nn_config.divideParam.testRatio  = 0.0;
    
    pos_weight = weights(rem(i-1,n_w)+1);
    weight = ones(size(data.y_train)); 
    weight(data.y_train == 1) = pos_weight; 
    
    fprintf('Model nn %d of %d train (weight=%f)...\n', i, n, pos_weight); 
    [cur_nn_model.model] = train(nn_config, data.x_train, ...
        data.y_train, [], [], weight);
    cur_nn_model.pos_weight = pos_weight;
    
    [cur_nn_model.score, ~] = nnetEval(cur_nn_model.model, data.x_train1, data.y_train1, ...
        data.x_test1, data.y_test1, score_fcn);
    
    % Saving
    if (cur_nn_model.score > pca_nn_model.score)
        fprintf('Saving nn model...\n');
        pca_nn_model = cur_nn_model;
        save pcaNnModelData.mat pca_nn_model;
        fprintf('Nn model saved...\n');
    else
        fprintf('Best %s is: %1.4f\n', score_fcn, pca_nn_model.score);
    end
    toc;
    fprintf('\n');
    model_hist{i} = cur_nn_model;    %#ok<SAGROW>
end