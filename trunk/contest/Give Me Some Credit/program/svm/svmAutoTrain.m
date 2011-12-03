function [svm_model, svm_auc] = svmAutoTrain(X_train, y_train, X_val, y_val, X_test, y_test, c_values, gamma_values)

    fprintf('Finding SVM params...\n');
    n_train = size(y_train, 1);
    n_find_params = 10000;
    if (n_find_params > n_train)
        n_find_params = n_train;
    end
    n_actual_train = 100000;
    if (n_actual_train > n_train)
        n_actual_train = n_train;
    end

    [svm_c, svm_gamma] = svmFindBestParams(X_train(1:n_find_params, :), y_train(1:n_find_params, :), ...
        X_val, y_val, c_values, gamma_values);
    fprintf('Params selected: C = %8.8f, gamma = %5.10f\n', ...
                svm_c(1), svm_gamma(1));

    fprintf('Training SVM params...\n');

    [svm_model] = svmTrainWrapper(X_train(1:n_actual_train, :), y_train(1:n_actual_train, :), svm_c(1), svm_gamma(1));
    [~, ~, ~, y_train_auc] = svmPredictWrapper(X_train, y_train, svm_model);
    [~, ~, ~, y_val_auc] = svmPredictWrapper(X_val, y_val, svm_model);
    [~, ~, ~, y_test_auc] = svmPredictWrapper(X_test, y_test, svm_model);
    fprintf('AUC: train auc = %1.4f, val auc = %1.4f, test auc = %1.4f\n', y_train_auc, y_val_auc, y_test_auc);
    svm_auc = min([y_train_auc y_val_auc y_test_auc]);

end