clear; clc;

%% loading values
tic;

if (~exist('parsedData.mat', 'file'))
    runParseDataMain;
end
load parsedData.mat;

rng(6677);

%% running kmeans
K = 4;
fprintf('Runing K means (k=%d) to cluster similar data\n\n', K);
x_train_all = [data.x_train1; data.x_test1; data.x_train2; data.x_test2];
y_train_all = [data.y_train1; data.y_test1; data.y_train2; data.y_test2];

[centroids, tr_idx] = runkMeans(x_train_all, K, 100, data.is_discrete);
sub_idx = findClosestCentroids(data.x_submit, centroids, data.is_discrete);

%% show centroi stats
centroids_stats = (1:K)';
for i=1:K
    centroids_stats(i,2) = sum(tr_idx == i);
    centroids_stats(i,3) = sum(sub_idx == i);
end
fprintf('\n\tCentroid\tTrain count\t\tSubmit count:\n\n');
disp(centroids_stats);

%% saving data
train_split_ratio = {0.4, 0.1, 0.4, 0.1};
for i=1:K
    kdata.centroid = centroids(i, :);
    [kdata.x_train1, kdata.y_train1, kdata.x_test1, kdata.y_test1, ...
        kdata.x_train2, kdata.y_train2, kdata.x_test2, kdata.y_test2] = split(x_train_all(tr_idx==i, :), ...
        y_train_all(tr_idx==i, :), train_split_ratio{:});
    kdata.x_submit = data.x_submit(sub_idx==i, :);
    cldata.k{i} = kdata;
end

save clusterData.mat cldata;

toc;
                                  



