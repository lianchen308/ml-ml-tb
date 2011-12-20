clear; clc;

%% loading values
tic;

if (~exist('rawData.mat', 'file'))
    runBuildRawDataMain;
end
load rawData.mat;
data = raw_data;


%% shuffling data
fprintf('Shuffling train data...\n');
rng(46793);
[data.x_train rnd_idx] = shuffle(data.x_train);
data.y_train = data.y_train(rnd_idx);
train_split_ratio = {0.4, 0.1, 0.4, 0.1};
[data.x_train1, data.y_train1, data.x_test1, data.y_test1, ...
    data.x_train2, data.y_train2, data.x_test2, data.y_test2] = split(data.x_train, ...
    data.y_train, train_split_ratio{:});
fprintf('Shuffle positive ratio: train1=%3.2f, test1=%3.2f, train2=%3.2f, test2=%3.2f\n', ...
    posratio(data.y_train1)*100, posratio(data.y_test1)*100, ...
    posratio(data.y_train2)*100, posratio(data.y_test2)*100);


%% normalizing data
fprintf('Normalizing data...\n');
[data.x_train, data.x_mu, data.x_std] = fnorm(data.x_train);
[data.x_submit] = fnorm(data.x_submit, data.x_mu, data.x_std);

fprintf('Normalization sumary...\n');
fprintf('\tDimension\tMean\tStd\t\t\tmax\t\tmin\t\t(norm training):\n');
disp([(1:size(data.x_train, 2))'  nanmean(data.x_train)' nanstd(data.x_train)' ...
    nanmax(data.x_train)' nanmin(data.x_train)']);
fprintf('\tDimension\tMean\tStd\t\t\tmax\t\tmin\t\t(norm submit):\n');
disp([(1:size(data.x_submit,2))'  nanmean(data.x_submit)' nanstd(data.x_submit)' ...
    nanmax(data.x_submit)' nanmin(data.x_submit)']);


%% estimate missing
k = 15;
fprintf('Estimating missing values by knn (k=%d)...\n', k);
data.x_train = knnimputeext(data.x_train, k, data.is_discrete);
data.x_submit = knnimputeext(data.x_submit, k, data.is_discrete);
fprintf('Missing values estimated...\n');

%% splitting training
 
fprintf('Splitting training data...\n');
[data.x_train1, data.y_train1, data.x_test1, data.y_test1, ...
    data.x_train2, data.y_train2, data.x_test2, data.y_test2] = split(data.x_train, ...
    data.y_train, train_split_ratio{:});
data = rmfield(data,'x_train');
data = rmfield(data,'y_train');


%% Finishing
fprintf('Saving data...\n');
save parsedData.mat data;

fprintf('Data parsed and saved\n\n');
toc;