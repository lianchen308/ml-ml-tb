% function [y_pred, y_acc, y_prob, y_auc] = svmpredictw(model, X, y)
function [y_pred, y_acc, y_prob, y_auc] = svmpredictw(model, X, y)
 
    if (~exist('y', 'var') || isempty(y))
        y = zeros(size(X,1), 1);
    end
    [~, y_acc, y_prob] = libsvmpredict(y, X, model, '-b 1');
    y_prob = y_prob(:,2);
    y_acc = y_acc(1);
    y_auc = aucscore(y, y_prob);
    y_pred = (y_prob - 0.5)*2;
    
end