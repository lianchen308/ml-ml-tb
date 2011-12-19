% [y_test_auc, y_test_acc] = knneval(knn_model, X_test, y_test, score_fcn)
function [y_test_score, y_test_acc] = knneval(knn_model, X_test, y_test, score_fcn)

    if (~exist('score_fcn', 'var') || isempty(score_fcn))
        score_fcn = 'aucscore';
    end
    
    fprintf('Evaluating knn...\n');
    [y_test_pred, ~, y_test_acc, y_test_score] = knnpredict(knn_model, X_test, y_test, score_fcn);
    [y_test_negacc, y_test_posacc] = classaccuracy(y_test, y_test_pred);
    fprintf('Test result: acc = %1.4f, neg acc = %1.4f, pos acc = %1.4f, %s = %1.4f\n', ...
        y_test_acc, y_test_negacc, y_test_posacc, score_fcn, y_test_score);
end