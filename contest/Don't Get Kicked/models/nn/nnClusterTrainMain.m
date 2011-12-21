clear; clc;
fprintf('Loading data...\n');


load('../../data/clusterData.mat');

for i=1:length(cldata.k)
    nn_cluster_model.k{i}.score = -1;
end

if (exist('nnClusterModelData.mat', 'file'))
    load nnClusterModelData.mat;
end

k = 1;
kdata = cldata.k{k};

% merging train data
data.x_train = [kdata.x_train1' kdata.x_test1'];
data.y_train = [kdata.y_train1' kdata.y_test1'];
data.x_test = [kdata.x_train2' kdata.x_test2'];
data.y_test = [kdata.y_train2' kdata.y_test2'];
data.x_all = [data.x_train data.x_test];
data.y_all = [data.y_train data.y_test];

n = 500;
model_hist = {};

n_w = 15;
weights = linspace(1, negposratio(data.y_train)*1.5, n_w);

score_fcn = 'giniscore';

x_nn_train = data.x_all;
y_nn_train = data.y_all;
    
for i=1:n
    % Training
    tic;
    nn_config = newff(minmax(data.x_all), minmax(data.y_all), ...
        50, {'tansig', 'tansig', 'tansig', 'tansig'}, 'trainlm');
    nn_config.trainParam.epochs = 200;
    nn_config.trainParam.max_fail = 40;
    nn_config.trainParam.min_grad = 1e-30;
    nn_config.divideFcn = 'divideblock';
    nn_config.divideParam.trainRatio = 0.5; 
    nn_config.divideParam.valRatio   = 0.5;
    nn_config.divideParam.testRatio  = 0.0;
    
    pos_weight = weights(rem(i-1,n_w)+1);
    weight = ones(size(y_nn_train)); 
    weight(y_nn_train == 1) = pos_weight; 
    
    fprintf('Model nn %d of %d train (weight=%f)...\n', i, n, pos_weight); 
    [cur_nn_model.model] = train(nn_config, x_nn_train, ...
       y_nn_train, [], [], weight);
    cur_nn_model.pos_weight = pos_weight;
    
    [cur_nn_model.score, ~] = nnetEval(cur_nn_model.model, data.x_train, data.y_train, ...
        data.x_test, data.y_test, score_fcn);
    
    % Saving
    if (cur_nn_model.score > nn_cluster_model.k{k}.score)
        fprintf('Saving nn model...\n');
        nn_cluster_model.k{k} = cur_nn_model;
        save nnClusterModelData.mat nn_cluster_model;
        fprintf('Nn model saved...\n');
    else
        fprintf('Best %s is: %1.4f\n', score_fcn, nn_cluster_model.k{k}.score);
    end
    toc;
    fprintf('\n');
    model_hist{i} = cur_nn_model;    %#ok<SAGROW>
end