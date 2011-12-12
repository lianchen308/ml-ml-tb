clear; clc; close all;

fprintf('Loading data');
load ../data/binaryData.mat;
[y_nnetboost, y_svm, y_rf] = loadModelData( X );
    
fprintf('Opening graphs');

viewdata(y_nnetboost, y_svm, y_train2, ...
    'nnet adaboost', 'svm', 'svm x nnet adaboost');

viewdata(y_svm, y_rf, y_train2, ...
    'nnet adaboost', 'random forest', 'nnet adaboost x random forest');

viewdata(y_rf, y_nnetboost, y_train2, ...
    'svm', 'random forest', 'svm x random forest');