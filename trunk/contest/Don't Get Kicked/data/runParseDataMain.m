clear; clc;

tic;

if (~exist('rawData.mat', 'file'))
    runBuildRawDataMain;
end
load rawData.mat;
data = raw_data;

fprintf('Shuffling train data...\n');
rng(48546);
[data.x_train rnd_idx] = shuffle(data.x_train);
data.y_train = data.y_train(rnd_idx);

fprintf('Normalizing data...\n');
[data.x_train, data.x_mu, data.x_std] = fnorm(data.x_train);
[data.x_submit] = fnorm(data.x_submit, data.x_mu, data.x_std);

fprintf('Normalization sumary...\n');
fprintf('\tDimension\tMean\tStd\t\t(training):\n');
disp([(1:size(data.x_train, 2))'  nanmean(data.x_train)' nanstd(data.x_train)']);
fprintf('\tDimension\tMean\tStd\t\t(submit):\n');
disp([(1:size(data.x_submit,2))'  nanmean(data.x_submit)' nanstd(data.x_submit)']);

k = 15;
fprintf('Estimating missing values by knn (k=%d)...\n', k);
%X_all_est = knnimputeext(X_all_norm, 200);

fprintf('Missing values estimated...\n');
% 
% fprintf('Splitting training and submit data...\n');
% 
% % Training: train, validation and test
% n_train = length(y_train);
% n_cut1 = round(0.3*n_train);
% n_cut2 = n_cut1 + round(0.3*n_train);
% n_cut3 = n_cut2 + round(0.2*n_train);
% X_train = X_all_est(1:n_train, :);
% 
% X_train1     = X_train(1:n_cut1, :);
% y_train1     = y_train(1:n_cut1, :);
% X_train1_nan = find(isnan(X_all_norm(1:n_cut1, :)));
% X_train2     = X_train(n_cut1+1:n_cut2, :);
% y_train2     = y_train(n_cut1+1:n_cut2, :);
% X_train2_nan = find(isnan(X_all_norm(n_cut1+1:n_cut2, :)));
% 
% X_val      = X_train(n_cut2+1:n_cut3, :);
% y_val      = y_train(n_cut2+1:n_cut3, :);
% X_val_nan  = find(isnan(X_all_norm(n_cut2+1:n_cut3, :)));
% 
% X_test     = X_train(n_cut3+1:n_train, :);
% y_test     = y_train(n_cut3+1:n_train, :);
% X_test_nan = find(isnan(X_all_norm(n_cut3+1:n_train, :)));
% 
% % Submission data
% X_submit     = X_all_est(n_train + 1 : end, :);
% X_submit_nan = find(isnan(X_all_est(n_train + 1 : end, :)));
% 



fprintf('Saving data...\n');
save parsedData.mat data;

fprintf('Data parsed and saved\n\n');
toc;