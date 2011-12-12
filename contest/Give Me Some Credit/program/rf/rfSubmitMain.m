clc; clear;
fprintf('Loading submission data...\n');
% loading data
load binaryRfModelData.mat;
load('../data/binarySubmitData.mat');

% Saving submit
fprintf('Running random forest model on submission data...\n');
[~, y_submit_prob] = rfpredict(rf_model, X_submit);
fprintf('Saving random forest submission data...\n');
dlmwrite('csvSubmitRfData.csv',[(1:length(y_submit_prob))' y_submit_prob], ...
    'delimiter',',','precision',6);
fprintf('Random forest saved...\n');
