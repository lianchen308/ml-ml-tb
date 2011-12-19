clear; clc;
fprintf('Loading data...\n');

knn_model.score = -1;
load('../../data/parsedData.mat');

if (exist('knnModelData.mat', 'file'))
    load knnModelData.mat; 
end

score_fcn = 'giniscore';
% Training
fprintf('Model Knn train...\n');
k_n = round(logspace(2, 3, 10));
for k =k_n
    [curr_knn_model.model] = knntrain(data.x_train1, data.y_train1, k);

    tic;
    fprintf('K=%d: ', k);
    [curr_knn_model.score, ~] = knneval(curr_knn_model.model, ...
        data.x_test1, data.y_test1, score_fcn);

    % Saving
    if (curr_knn_model.score > knn_model.score)
        fprintf('Saving knn model...\n');
        [curr_knn_model.model] = knntrain([data.x_train1; data.x_test1], ...
            [data.y_train1; data.y_test1], k);
        knn_model = curr_knn_model;
        save knnModelData.mat knn_model;

        fprintf('Knn saved...\n');
    else
        fprintf('Best %s is: %1.4f\n', score_fcn, knn_model.score);
    end
    toc;
    fprintf('\n');
end