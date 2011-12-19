% function [y_pred, y_acc, y_prob, y_score] = svmpredictw(model, X, y, score_fcn)
function [y_pred, y_acc, y_prob, y_score] = svmpredictw(model, X, y, score_fcn)
 
    if (~exist('y', 'var') || isempty(y))
        y = zeros(size(X,1), 1);
    end
    if (~exist('score_fcn', 'var') || isempty(score_fcn))
		score_fcn = 'aucscore';
    end
    
    [~, y_acc, y_prob] = libsvmpredict(y, X, model, '-b 1');
    y_prob = y_prob(:,2);
    y_acc = y_acc(1);
    y_pred = (y_prob - 0.5)*2;
    
    y_score = feval(score_fcn, y, y_prob);
    
end