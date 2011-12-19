clear; clc;

svm_model.score = -1;
load('../../data/parsedData.mat');

if (exist('svmModelData.mat', 'file'))
    load svmModelData.mat;
end


c_values = logspace(1, 7, 10); gamma_values = logspace(-8, -0, 10); % -> search!
n_find_params = 5000;
n_actual_train= 45000;
opt.score_fcn = 'giniscore';
opt.pos_weights = 7.928269;

% Training
tic;
[curr_model.model, curr_model.score, curr_model.c, curr_model.gamma] = svmgridsearch( ...
    data.x_train1, data.y_train1, data.x_test1, data.y_test1, ...
    c_values, gamma_values, opt, n_find_params, n_actual_train);

[~, ~, ~, y_train1_score] = svmpredictw(curr_model.model, ...
    data.x_train1, data.y_train1, opt.score_fcn);
fprintf('%s: train1 = %1.4f\n', opt.score_fcn, y_train1_score);

curr_model.score = min(y_train1_score, curr_model.score);

% Saving
if (curr_model.score > svm_model.score)
    fprintf('Saving svm model...\n');
    svm_model = curr_model;
    save svmModelData.mat svm_model;
    fprintf('Svm model saved...\n');
else
    fprintf('Best auc is: %1.4f\n', svm_model.score);
end
toc;
fprintf('\n');