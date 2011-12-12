clc; clear;
fprintf('Loading submission data...\n');
% loading data
load binaryMixBoostModelData; 
load binarySubmitMixData.mat;

% Saving submit
fprintf('Running mixed adaboost model on submission data...\n');
[y_submit_out] = mabclassify(nnet_mix_learners, nnet_mix_weights, X_submit_mix);
[y_submit_pred, y_submit_prob] = predlabel(y_submit_out);
fprintf('Saving mixed adaboost submission data...\n');
dlmwrite('csvSubmitMixBoostData.csv',[(1:length(y_submit_prob))' y_submit_prob], ...
    'delimiter',',','precision',6);
fprintf('mixed adaboost saved...\n');
