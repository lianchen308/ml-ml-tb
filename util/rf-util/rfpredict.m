function [y_pred, y_prob, acc, auc] = rfpredict(model, X, y)

    [~, y_prob] = predict(model, X);
    
    y_prob = y_prob(:, 2);
    
    y_pred = zeros(size(y_prob));
    y_pred(y_prob >= 0.5) = 1;
    y_pred(y_prob < 0.5) = -1;
    
    if (~exist('y', 'var') || isempty(y))
        acc = -1;
        auc = -1;
    else
        acc = (length(find(y_pred == y))/length(y));
        auc = aucscore(y, y_pred);
    end

end