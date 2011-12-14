clc; clear; 

tic;
fprintf('Loading data...\n');
load ../data/binarySubmitData.mat;
load ../data/binaryData.mat;


fprintf('Loading submit data...\n');
[X_submit_nnboost, X_submit_svm, X_submit_rf, X_submit_knn ...
    X_submit_svmru, X_submit_nnet, X_submit_gaussian] = loadModelData( X_submit );
X_submit_mix = [X_submit X_submit_nnboost X_submit_svm X_submit_rf X_submit_knn ...
    X_submit_svmru, X_submit_nnet, X_submit_gaussian];

fprintf('Loading train1 data...\n');
[X_train1_nnboost, X_train1_svm, X_train1_rf, X_train1_knn ...
    X_train1_svmru, X_train1_nnet, X_train1_gaussian] = loadModelData( X_train1 );
X_train1_mix = [X_train1 X_train1_nnboost X_train1_svm X_train1_rf X_train1_knn ...
    X_train1_svmru, X_train1_nnet, X_train1_gaussian];

fprintf('Loading train2 data...\n');
[X_train2_nnboost, X_train2_svm, X_train2_rf, X_train2_knn ...
    X_train2_svmru, X_train2_nnet, X_train2_gaussian] = loadModelData( X_train2 );
X_train2_mix = [X_train2 X_train2_nnboost X_train2_svm X_train2_rf X_train2_knn ...
    X_train2_svmru, X_train2_nnet, X_train2_gaussian];

fprintf('Loading validation data...\n');
[X_val_nnboost, X_val_svm, X_val_rf, X_val_knn ...
    X_val_svmru, X_val_nnet, X_val_gaussian] = loadModelData( X_val );
X_val_mix = [X_val X_val_nnboost X_val_svm X_val_rf, X_val_knn ...
    X_val_svmru, X_val_nnet, X_val_gaussian];

fprintf('Loading test data...\n');
[X_test_nnboost, X_test_svm, X_test_rf, X_test_knn ...
    X_test_svmru, X_test_nnet, X_test_gaussian] = loadModelData( X_test );
X_test_mix = [X_test X_test_nnboost X_test_svm X_test_rf X_test_knn ...
    X_test_svmru, X_test_nnet, X_test_gaussian];

fprintf('Saving all data...\n');
save binaryMixData.mat X_train1_mix X_train2_mix X_val_mix X_test_mix;
save binarySubmitMixData.mat X_submit_mix;
fprintf('Done!\n');
toc;