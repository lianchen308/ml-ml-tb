clear; clc;
fprintf('Loading data...\n');

mix_rf_model_auc = -1;
load ../data/binaryData.mat;
load binaryMixData.mat;


X_train1 = X_train1_mix(:, 11:end);
X_train2 = X_train2_mix(:, 11:end);
X_val    = X_val_mix(:, 11:end);
X_test   = X_test_mix(:, 11:end);

X_prev = [X_train1; X_val];
y_prev = [y_train1; y_val];
X_train = [X_train2; ];
y_train = [y_train2; ];

if (exist('binaryMixRfModelData.mat', 'file'))
    load binaryMixRfModelData.mat;
    clear mix_rf_model;
end

sampsize = [1000 2000 3000 5000];
n_sampsize = length(sampsize);


trees = [500 1000 2000 4000 8000 12000];
n_trees = length(trees);

mtry = [1 2 3];
n_mtrys = length(mtry);


n = 300;
i_samplesize = 1;
i_tree = 1;
i_dims = 1;

for i=1:n
    tic;
    fprintf('Model random forest %d of %d train...\n\n', i, n);
    fprintf('Sample size = %d, ntree = %d, mtry = %d...\n', sampsize(i_samplesize), trees(i_tree), mtry(i_dims));
    
    extra_options.sampsize = sampsize(i_samplesize);
    if (extra_options.sampsize > 0.638*length(y_train))
        extra_options.replace = 0; 
    end
    [cur_rf_model] = classRF_train(X_train, y_train, trees(i_tree), mtry(i_dims), extra_options);

    fprintf('OOB error %3.2f%%, negative: %3.2f%%, positive: %3.2f%%...\n\n', cur_rf_model.errtr(end,1)*100, ...
        cur_rf_model.errtr(end,2)*100, cur_rf_model.errtr(end,3)*100);
    
    
    [~, prev_auc, ~, test_auc, ~, train_auc, ~, all_auc] = rfeval(cur_rf_model, X_train2, y_train2, X_test, y_test, ...
        X_prev, y_prev);
    
    curr_rf_model_auc = min([prev_auc test_auc train_auc all_auc]);
    
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