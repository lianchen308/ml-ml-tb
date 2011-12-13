clear; clc;
fprintf('Loading data...\n');

mix_rf_model_auc = -1;
load ../data/binaryData.mat;
load binaryMixData.mat;
load binaryMixBoostModelData.mat;

[X_train1_nnet] = mabclassify(nnet_mix_learners, nnet_mix_weights, X_train1_mix, 1);
[X_train2_nnet] = mabclassify(nnet_mix_learners, nnet_mix_weights, X_train2_mix, 1);
[X_val_nnet] = mabclassify(nnet_mix_learners, nnet_mix_weights, X_val_mix, 1);
[X_test_nnet] = mabclassify(nnet_mix_learners, nnet_mix_weights, X_test_mix, 1);

X_train1 = [X_train1_mix X_train1_nnet];
X_train2 = [X_train2_mix X_train2_nnet];
X_val = [X_val_mix X_val_nnet];
X_test = [X_test_mix X_test_nnet];

if (exist('binaryMixRfModelData.mat', 'file'))
    load binaryMixRfModelData.mat; 
end

n = 5;
for i=1:n
    % Training
    tic;
    fprintf('Mixed random forest model %d of %d train...\n', i, n);
    
    extra_options.sampsize = 2500;
    %extra_options.importance = 1;
    [cur_rf_model] = classRF_train(X_train2, y_train2, 10000, 3, extra_options);

    [curr_rf_model_auc, ~] = rfeval(cur_rf_model, X_train2, y_train2, ... 
        X_val, y_val, X_test, y_test);

    % Saving
    if (curr_rf_model_auc > mix_rf_model_auc)
        fprintf('Saving random forest model...\n');
        mix_rf_model = cur_rf_model;
        mix_rf_model_auc = curr_rf_model_auc;
        save binaryMixRfModelData.mat mix_rf_model mix_rf_model_auc;

        fprintf('random forest model saved...\n');
    else
        fprintf('Best auc is: %1.4f\n', mix_rf_model_auc);
    end
    toc;
    fprintf('\n');
end