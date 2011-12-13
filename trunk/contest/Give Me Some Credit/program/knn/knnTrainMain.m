clear; clc;
fprintf('Loading data...\n');

knn_model_auc = -1;
load ../data/binaryData.mat;
if (exist('binaryKnnModelData.mat', 'file'))
    load binaryKnnModelData.mat; 
end


% Training
fprintf('Model Knn train...\n');
k_n = round(logspace(2, 2.5, 10));
for k =k_n
    [cur_knn_model] = knntrain([X_train1; X_val], [y_train1; y_val], k);

    tic;
    fprintf('K=%d: ', k);
    [curr_knn_model_auc, ~] = knneval(cur_knn_model, X_test, y_test);

    % Saving
    if (curr_knn_model_auc > knn_model_auc)
        fprintf('Saving knn model...\n');
        knn_model = cur_knn_model;
        knn_model_auc = curr_knn_model_auc;
        save binaryKnnModelData.mat knn_model knn_model_auc;

        fprintf('Knn saved...\n');
    else
        fprintf('Best auc is: %1.4f\n', knn_model_auc);
    end
    toc;
    fprintf('\n');
end