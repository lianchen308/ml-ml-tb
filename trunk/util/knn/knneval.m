% [y_test_auc, y_test_acc] = knneval(knn_model, X_test, y_test)
function [y_test_auc, y_test_acc] = knneval(knn_model, X_test, y_test)

    fprintf('Evaluating knn...\n');
    [~, ~, y_test_acc, y_test_auc] = knnpredict(knn_model, X_test, y_test);
    fprintf('Test result: acc = %1.4f, auc = %1.4f\n', y_test_acc, y_test_auc);

end