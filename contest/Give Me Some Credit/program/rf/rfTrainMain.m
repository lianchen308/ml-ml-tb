clear; clc;
fprintf('Loading data...\n');

rf_model_auc = -1;
load ../data/binaryData.mat;
if (exist('binaryRfModelData.mat', 'file'))
    load binaryRfModelData.mat; 
    clear rf_model;
end

sampsize = [2000 4000 8000];
n_sampsize = length(sampsize);

trees = [4000 8000 16000 32000];
n_trees = length(trees);

dims = [2 3 4];
n_dims = length(dims);

n = 1000;
i_samplesize = 1;
i_tree = 1;
i_dims = 1;
for i=1:n
    % Training
    tic;
    fprintf('Model random forest %d of %d train...\n', i, n);
    fprintf('Sample size = %d, ntrees = %d, dims = %d...\n', sampsize(i_samplesize), trees(i_tree), dims(i_dims));
    extra_options.sampsize = sampsize(i_samplesize);
    [cur_rf_model] = classRF_train([X_train1; X_val], [y_train1; y_val], trees(i_tree), dims(i_dims), extra_options);

    [curr_rf_model_auc, ~] = rfeval(cur_rf_model, X_train1, y_train1, X_val, y_val, ...
        X_test, y_test);
    
    [~, ~, y_train2_acc, y_train2_auc] = rfpredict(cur_rf_model, X_train2, y_train2);
    fprintf('Train2 result: acc = %1.4f, auc = %1.4f\n', y_train2_acc, y_train2_auc);

    curr_rf_model_auc = min(y_train2_auc, curr_rf_model_auc);
    % Saving
    if (curr_rf_model_auc > rf_model_auc)
        fprintf('Saving random forest model...\n');
        rf_model = cur_rf_model;
        rf_model_auc = curr_rf_model_auc;
        save binaryRfModelData.mat rf_model rf_model_auc;

        fprintf('random forest model saved...\n');
		clear rf_model;
    else
        fprintf('Best auc is: %1.4f\n', rf_model_auc);
    end
    toc;
    fprintf('\n');
    clear cur_rf_model;
    clear FUNCTIONS;
    
    i_dims = i_dims + 1;
    if (i_dims > n_dims)
        i_dims = 1;
        i_tree = i_tree + 1;
        if (i_tree > n_trees)
            i_tree = 1;
            i_samplesize = i_samplesize + 1;
            if (i_samplesize > n_sampsize)
                i_samplesize = 1;
            end
        end  
    end
end