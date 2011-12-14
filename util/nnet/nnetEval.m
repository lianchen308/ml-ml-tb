function [y_test_auc, y_test_acc] = nnetEval(nn_model, X_train, y_train, X_val, y_val, X_test, y_test)

    fprintf('Evaluating nnet...\n');
    [~, ~, y_train_acc, y_train_auc] = nnetPredict(nn_model, X_train, y_train);
    fprintf('Train result: acc = %1.4f, auc = %1.4f\n', y_train_acc, y_train_auc);
    [~, ~, y_val_acc, y_val_auc] = nnetPredict(nn_model, X_val, y_val);
    fprintf('Val result: acc = %1.4f, auc = %1.4f\n', y_val_acc, y_val_auc);
    [y_test_pred, ~, y_test_acc, y_test_auc] = nnetPredict(nn_model, X_test, y_test);
    fprintf('Test result: acc = %1.4f, auc = %1.4f\n', y_test_acc, y_test_auc);
    [y_test_negacc, y_test_posacc] = classaccuracy(y_test, y_test_pred);
    fprintf('Test acc: neg acc = %1.4f, pos acc = %1.4f\n', y_test_negacc, y_test_posacc);
end