% [y_test_acc, y_test_auc, y_val_acc, y_val_auc, y_train_acc, y_train_auc, y_all_acc, y_all_auc] = ...
%    rfeval(rf_model, X_train, y_train, X_val, y_val, X_test, y_test)
function [y_test_acc, y_test_auc, y_val_acc, y_val_auc, y_train_acc, y_train_auc, y_all_acc, y_all_auc] = ...
    rfeval(rf_model, X_train, y_train, X_val, y_val, X_test, y_test)

    fprintf('Evaluating random forest...\n');
    [y_train_pred, ~, y_train_acc, y_train_auc] = rfpredict(rf_model, X_train, y_train);
    fprintf('Train result: acc = %1.4f, auc = %1.4f\n', y_train_acc, y_train_auc);
    
    [y_val_pred, ~, y_val_acc, y_val_auc] = rfpredict(rf_model, X_val, y_val);
    fprintf('Val result: acc = %1.4f, auc = %1.4f\n', y_val_acc, y_val_auc);
    
    [y_test_pred, ~, y_test_acc, y_test_auc] = rfpredict(rf_model, X_test, y_test);
    [y_test_negacc, y_test_posacc] = classaccuracy(y_test, y_test_pred);
    fprintf('Test result: acc = %1.4f, neg acc = %1.4f, pos acc = %1.4f, auc = %1.4f\n', ...
        y_test_acc, y_test_negacc, y_test_posacc, y_test_auc);
    
    y_all = [y_train; y_val; y_test];
    y_all_pred = [y_train_pred; y_val_pred; y_test_pred];
    y_all_acc = accuracy(y_all, y_all_pred);
    y_all_auc = aucscore(y_all, y_all_pred);
    [y_all_negacc, y_all_posacc] = classaccuracy(y_all, y_all_pred);
    fprintf('All result: acc = %1.4f, neg acc = %1.4f, pos acc = %1.4f, auc = %1.4f\n', ...
        y_all_acc, y_all_negacc, y_all_posacc, y_all_auc);

end