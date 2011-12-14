clear; clc;
fprintf('Loading data...\n');

mix_rf_model_auc = -1;
load ../data/binaryData.mat;
load binaryMixData.mat;
load binaryMixBoostModelData.mat;

[X_train1_nnet] = mabclassify(nnet_mix_learners, nnet_mix_weights, X_train1_mix, 1);
[X_train2_nnet] = mabclassify(nnet_mix_learners, nnet_mix_weights, X_train2_mix, 1);
[X_val_nnet]    = mabclassify(nnet_mix_learners, nnet_mix_weights, X_val_mix, 1);
[X_test_nnet]   = mabclassify(nnet_mix_learners, nnet_mix_weights, X_test_mix, 1);

X_train1 = [X_train1_mix X_train1_nnet];
X_train2 = [X_train2_mix X_train2_nnet];
X_val    = [X_val_mix X_val_nnet];
X_test   = [X_test_mix X_test_nnet];

X_prev = [X_train1; X_val];
y_prev = [y_train1; y_val];

if (exist('binaryMixRfModelData.mat', 'file'))
    load binaryMixRfModelData.mat; 
end

sampsize = [30000];
n_sampsize = length(sampsize);

trees = [30000];
n_trees = length(trees);

mtry = 4;
n_mtrys = length(mtry);


n = 100;
i_samplesize = 1;
i_tree = 1;
i_dims = 1;
for i=1:n
    tic;
    fprintf('Model random forest %d of %d train...\n', i, n);
    fprintf('Sample size = %d, ntree = %d, mtry = %d...\n', sampsize(i_samplesize), trees(i_tree), mtry(i_dims));
    extra_options.sampsize = sampsize(i_samplesize);

    [cur_rf_model] = classRF_train(X_train2, y_train2, trees(i_tree), mtry(i_dims), extra_options);

    [curr_rf_model_auc, ~] = rfeval(cur_rf_model, X_train2, y_train2, X_prev, y_prev, ...
        X_test, y_test);
    
%     fprintf('Importance matrix...\n');
%     disp(cur_rf_model.importance);
    
    % Saving
    if (curr_rf_model_auc > mix_rf_model_auc)
        fprintf('Saving random forest model...\n');
        mix_rf_model = cur_rf_model;
        mix_rf_model_auc = curr_rf_model_auc;
        save binaryMixRfModelData.mat mix_rf_model mix_rf_model_auc;

        fprintf('random forest model saved...\n');
		clear rf_model;
    else
        fprintf('Best auc is: %1.4f\n', mix_rf_model_auc);
    end
    toc;
    fprintf('\n');
    clear cur_rf_model;
    clear FUNCTIONS;
    
    i_dims = i_dims + 1;
    if (i_dims > n_mtrys)
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