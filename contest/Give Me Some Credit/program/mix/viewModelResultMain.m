clear; clc; close all;

fprintf('Loading data...\n');
load ../data/binaryData.mat;
load binaryMixData.mat;

y_nnetboost = X_train2_mix(:, 11);
y_svm = X_train2_mix(:, 12);
y_rf = X_train2_mix(:, 13);
y_knn = X_train2_mix(:, 14);
    
fprintf('Opening graphs...\n');

viewdata(y_nnetboost, y_svm, y_train2, ...
    'nnet adaboost', 'svm', 'nnet adaboost x svm');

viewdata(y_svm, y_rf, y_train2, ...
    'svm', 'random forest', 'svm x random forest');

viewdata(y_rf, y_nnetboost, y_train2, ...
    'random forest', 'nnet adaboost', 'random forest x nnet adaboost');

viewdata(y_nnetboost, y_knn, y_train2, ...
    'nnet adaboost', 'knn', 'nnet adaboost x knn');

viewdata(y_svm, y_knn, y_train2, ...
    'svm', 'knn', 'svm x knn');

viewdata(y_rf, y_knn, y_train2, ...
    'random forest', 'knn', 'random forest x knn');

fprintf('Done!\n');