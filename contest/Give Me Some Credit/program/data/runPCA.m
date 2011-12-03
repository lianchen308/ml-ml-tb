clear; clc;

fprintf('Runing PCA\n');
load binaryData.mat; % X_train y_train X_val y_val X_test y_test;
load binarySubmitData.mat; % X_submit;
X_train = X_train'; y_train = y_train'; X_val = X_val'; y_val = y_val'; X_test = X_test'; y_test = y_test'; X_submit = X_submit';
var_retained = 0.99;

[~, Z_U, Z_S, Z_K, Z_cum_sigma] = pca([X_train; X_val; X_test; X_submit], var_retained);
Z_cum_sigma
Z_train = projectdata(X_train, Z_U, Z_K);
Z_val = projectdata(X_val, Z_U, Z_K);
Z_test = projectdata(X_test, Z_U, Z_K);
Z_submit = projectdata(X_submit, Z_U, Z_K);

save binaryPCAData.mat Z_train y_train Z_val y_val Z_test y_test;
save binaryPCASubmitData.mat Z_submit;

fprintf('Features reduced to %d features retaining %d%% of variance\n', Z_K, var_retained*100);
