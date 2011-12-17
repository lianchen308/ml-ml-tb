clear; clc;

tic;

if (~exist('rawData.mat', 'file'))
    runBuildRawDataMain;
end
load rawData.mat;
data = raw_data;

fprintf('Normalizing data...\n');
[data.x_train, data.x_mu, data.x_std] = fnorm(data.x_train);
[data.x_submit] = fnorm(data.x_submit, data.x_mu, data.x_std);

fprintf('Getting nan statistics...\n');
x_train_nans = isnan(data.x_train);
nan_per_col = sum(x_train_nans);

fprintf('Removing nan rows...\n');
x_train_nan_rows = sum(isnan(data.x_train),2);
x_train_non_nan = data.x_train;
x_train_non_nan(x_train_nan_rows > 0, :) = [];

fprintf('Injecting random nans based on statistics collected...\n');
n_non_nan = size(x_train_non_nan, 1);
x_train_inject_nan = x_train_non_nan;
nan_cols = find(nan_per_col > 0);
for col=nan_cols
    rand_idx = randperm(n_non_nan, nan_per_col(col));
    x_train_inject_nan(rand_idx, col) = NaN;
end
injected_nans = isnan(x_train_inject_nan);


% knn neighboors
knn_types =  repmat('c', 1, length(data.is_discrete));
dicrete_idx = find(data.is_discrete);
knn_types(dicrete_idx) = repmat('d', 1, length(dicrete_idx));
for k=[1 5 10 15 20 50 100 200]
    fprintf('Running Knn neighboors (k=%d)...\n', k);
    x_knnimpute{k} = knnimputeext(x_train_inject_nan, k, data.is_discrete); %#ok<SAGROW>
    x_knnimputeloss(k) = imputationLossMixed(x_train_non_nan, ...
        x_knnimpute{k}, injected_nans, knn_types); %#ok<SAGROW>
    fprintf('Knn neighboors (k=%d) loss: %f...\n', k, x_knnimputeloss(k));
end



fprintf('Done!\n\n');
toc;