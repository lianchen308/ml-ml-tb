% [y_test_auc, y_test_acc] = rfeval(rf_model, X_train, y_train, X_val, y_val, X_test, y_test)
function [y_test_auc, y_test_acc] = rfeval(rf_model, X_train, y_train, X_val, y_val, X_test, y_test)

    fprintf('Evaluating random forest...\n');
    [~, ~, y_train_acc, y_train_auc] = rfpredict(rf_model, X_train, y_train);
    fprintf('Train result: acc = %1.4f, auc = %1.4f\n', y_train_acc, y_train_auc);
    [~, ~, y_val_acc, y_val_auc] = rfpredict(rf_model, X_val, y_val);
    fprintf('Val result: acc = %1.4f, auc = %1.4f\n', y_val_acc, y_val_auc);
    [~, ~, y_test_acc, y_test_auc] = rfpredict(rf_model, X_test, y_test);
    fprintf('Test result: acc = %1.4f, auc = %1.4f\n', y_test_acc, y_test_auc);

end