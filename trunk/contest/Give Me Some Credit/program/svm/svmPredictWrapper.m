function [y_pred, y_acc, y_prob, y_auc] = svmPredictWrapper(X, y, model)
 
    [y_pred, y_acc, y_prob] = libsvmpredict(y, X, model, '-b 1');
    y_prob = y_prob(:,2);
    y_acc = y_acc(1);
    y_auc = aucscore(y, y_prob);

end