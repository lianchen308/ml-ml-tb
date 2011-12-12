clear; clc;
fprintf('Loading data...\n');

rf_model_auc = -1;
load ../data/binaryData.mat;
if (exist('binaryRfModelData.mat', 'file'))
    load binaryRfModelData.mat; 
end

n = 5;
for i=1:n
    % Training
    tic;
    fprintf('Model random forest %d of %d train...\n', i, n);
    
    extra_options.sampsize = 4000;
    [cur_rf_model] = classRF_train([X_train1; X_val], [y_train1; y_val], 5000, 3, extra_options);

    [curr_rf_model_auc, ~] = rfeval(cur_rf_model, X_train1, y_train1, X_val, y_val, ...
        X_test, y_test);

    % Saving
    if (curr_rf_model_auc > rf_model_auc)
        fprintf('Saving random forest model...\n');
        rf_model = cur_rf_model;
        rf_model_auc = curr_rf_model_auc;
        save binaryRfModelData.mat rf_model rf_model_auc;

        fprintf('random forest model saved...\n');
    else
        fprintf('Best auc is: %1.4f\n', rf_model_auc);
    end
    toc;
    fprintf('\n');
end