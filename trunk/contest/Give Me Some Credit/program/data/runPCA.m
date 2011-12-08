var_retained = 0.99;
fprintf('Runing PCA to achieve at least %3.2f%% of variance\n', var_retained*100);
load binaryData.mat; % X_train y_train X_val y_val X_test y_test;
load binarySubmitData.mat; % X_submit;

[~, Z_U, Z_S, Z_K, Z_cum_sigma] = pca([X_train; X_val; X_test; X_submit], var_retained);
Z_train = projectdata(X_train, Z_U, Z_K);
Z_val = projectdata(X_val, Z_U, Z_K);
Z_test = projectdata(X_test, Z_U, Z_K);
Z_submit = projectdata(X_submit, Z_U, Z_K);

save binaryPCAData.mat Z_train y_train Z_val y_val Z_test y_test;
save binaryPCASubmitData.mat Z_submit;

Z_cum_sigma(:, 2) = 100*Z_cum_sigma(:, 2);
fprintf('Features reduced to %d features retaining %3.2f%% of variance.\n', Z_K, Z_cum_sigma(Z_K, 2));
fprintf('Variance by dimension:\n');
disp(Z_cum_sigma);
