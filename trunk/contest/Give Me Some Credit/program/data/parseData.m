clear; clc;

tic;
fprintf('Loading data...\n');

% loads train data
rng(398475);
train_data = shuffle(csvread('../../raw data/cs-training-parsed.csv'));
y_train = train_data(:, 1);
y_train(y_train == 0) = -1;
X_train = train_data(:, 2:11);
clear train_data;

% loads submit data
X_submit = csvread('../../raw data/cs-test-parsed.csv');

fprintf('Normalizing data and replacing outliers...\n');
% puts all together for input treatment
X_all = [X_train; X_submit];
X_all(:, 1) = logsat(X_all(:, 1), 10, 100);
X_all(:, 4) = logsat(X_all(:, 4), 5);
% normalization
[X_all_norm, X_mu, X_std] = fnorm(X_all);

fprintf('Estimating missing values...\n');
% applies nearest neighboor to estimate missing
X_all_est = knnimputeext(X_all_norm, 100);
fprintf('Missing values estimated...\n');

fprintf('Splitting training and submit data...\n');

% Training: train, validation and test
n_train = length(y_train);
n_cut1 = round(0.3*n_train);
n_cut2 = n_cut1 + round(0.3*n_train);
n_cut3 = n_cut2 + round(0.2*n_train);
X_train = X_all_est(1:n_train, :);

X_train1     = X_train(1:n_cut1, :);
y_train1     = y_train(1:n_cut1, :);
X_train1_nan = find(isnan(X_all_norm(1:n_cut1, :)));
X_train2     = X_train(n_cut1+1:n_cut2, :);
y_train2     = y_train(n_cut1+1:n_cut2, :);
X_train2_nan = find(isnan(X_all_norm(n_cut1+1:n_cut2, :)));

X_val      = X_train(n_cut2+1:n_cut3, :);
y_val      = y_train(n_cut2+1:n_cut3, :);
X_val_nan  = find(isnan(X_all_norm(n_cut2+1:n_cut3, :)));

X_test     = X_train(n_cut3+1:n_train, :);
y_test     = y_train(n_cut3+1:n_train, :);
X_test_nan = find(isnan(X_all_norm(n_cut3+1:n_train, :)));

% Submission data
X_submit     = X_all_est(n_train + 1 : end, :);
X_submit_nan = find(isnan(X_all_est(n_train + 1 : end, :)));

fprintf('Normalization sumary...\n');
fprintf('\tDimension\tMean\tStd\t\t(training):\n');
disp([(1:size(X_train, 2))'  mean(X_train)' std(X_train)']);
fprintf('\tDimension\tMean\tStd\t\t(submit):\n');
disp([(1:size(X_submit,2))'  mean(X_submit)' std(X_submit)']);


fprintf('Saving data...\n');
save binaryData.mat X_train1 y_train1 X_train1_nan X_train2 y_train2 X_train2_nan X_val y_val X_val_nan X_test y_test X_test_nan X_mu X_std;
save binarySubmitData.mat X_submit X_submit_nan X_mu X_std;
fprintf('Data parsed and saved\n\n');


%runPCA;
toc;