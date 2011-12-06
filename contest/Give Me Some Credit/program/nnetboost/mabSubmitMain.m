clc; clear;
fprintf('Loading submission data...\n');
% loading data
load binaryAdaboostModelData.mat; % learners weights
load('../data/binarySubmitData.mat'); % X_submit



% Saving submit
fprintf('Running adaboost model on submission data...\n');
[y_submit_out] = mabclassify(learners, weights, X_submit);
[y_submit_pred, y_submit_prob] = predlabel(y_submit_out);
fprintf('Saving adaboost submission data...\n');
dlmwrite('csvSubmitAdaboostData.csv',[(1:length(y_submit_prob))' y_submit_prob'], ...
    'delimiter',',','precision',6);
fprintf('Adaboost saved...\n');
