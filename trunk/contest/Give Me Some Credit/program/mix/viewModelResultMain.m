clear; clc; close all;

fprintf('Loading data...\n\n');
load ../data/binaryData.mat;
load binaryMixData.mat;
plot = 0;

if (plot)   
    fprintf('Opening graphs...\n\n');
end

data.nnetboost = X_train2_mix(:, 11);
data.svm = X_train2_mix(:, 12);
data.rf = X_train2_mix(:, 13);
data.knn = X_train2_mix(:, 14);
data.svmru = X_train2_mix(:, 15);
data.nnet = X_train2_mix(:, 16);
data.gaussian = X_train2_mix(:, 17);
[consolidated_data, data_score] = consolidatealldata(data, y_train2, plot);

[~, ~, ~, ~, all_variance] = pca(X_train2_mix(:, 11:end), 1);
all_variance(:, 2) = all_variance(:, 2)*100;

fid = fopen(sprintf('modelscore.%s.txt', datestr(now, 'yyyy-mm-dd.HH-MM')),'a');

fprintf2(fid, 'Model score (AUC):\n\n');
fprintf2(fid, [struct2str(data_score) '\n']);
fprintf2(fid, 'PCA variance per feature projection:\n\n');
fprintf2(fid, '\t%d\t%3.2f\n', all_variance');
fprintf2(fid, '\nData variance:\n\n');
for i=1:length(consolidated_data)
    data = consolidated_data{i};
    fprintf2(fid, '\t%s x %s: %3.2f%%%%\n', data.x1_name, data.x2_name, ...
        data.variance*100); 
end

fclose(fid);

fprintf('\n\nDone!\n');