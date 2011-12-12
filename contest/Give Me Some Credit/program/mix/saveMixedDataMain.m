clc; clear; 

fprintf('Loading data...\n');
load ../data/binarySubmitData.mat;
load ../data/binaryData.mat;


fprintf('Saving submit data...\n');
[X_submit_nnboost, X_submit_svm, X_submit_rf] = loadModelData( X_submit );
X_submit_mix = [X_submit X_submit_nnboost X_submit_svm X_submit_rf];

fprintf('Saving train1 data...\n');
[X_train1_nnboost, X_train1_svm, X_train1_rf] = loadModelData( X_train1 );
X_train1_mix = [X_train1 X_train1_nnboost X_train1_svm X_train1_rf];

fprintf('Saving train2 data...\n');
[X_train2_nnboost, X_train2_svm, X_train2_rf] = loadModelData( X_train2 );
X_train2_mix = [X_train2 X_train2_nnboost X_train2_svm X_train2_rf];

fprintf('Saving validation data...\n');
[X_val_nnboost, X_val_svm, X_val_rf] = loadModelData( X_val );
X_val_mix = [X_val X_val_nnboost X_val_svm X_val_rf];

fprintf('Saving test data...\n');
[X_test_nnboost, X_test_svm, X_test_rf] = loadModelData( X_test );
X_test_mix = [X_test X_test_nnboost X_test_svm X_test_rf];

save binaryMixData.mat X_train1_mix X_train2_mix X_val_mix X_test_mix;
save binarySubmitMixData.mat X_submit_mix;