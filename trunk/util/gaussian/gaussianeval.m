% [y_test_auc, y_test_acc] = gaussianeval(gaussian_model, X_test, y_test)
function [y_test_auc, y_test_acc] = gaussianeval(gaussian_model, X_test, y_test)

    fprintf('Evaluating gaussian...\n');
    [~, ~, y_test_acc, y_test_auc] = gaussianpredict(gaussian_model, X_test, y_test);
    fprintf('Test result: acc = %1.4f, auc = %1.4f\n', y_test_acc, y_test_auc);

end