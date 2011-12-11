% function [y_pred, y_prob, y_acc] = predlabel(y, y_out)
function [y_pred, y_prob, acc] = predlabel(y, y_out)
    if (~exist('y_out', 'var') || isempty(y_out))
        y_out = y;
        acc = -1;
    end
    y_pred = zeros(size(y));
    y_pred(y_out >= 0) =  1;
    y_pred(y_out < 0)  = -1;
    
    y_prob = zeros(size(y));
    y_prob(y_out >= 0) =  y_out(y_out >= 0)/max([max(y_out) 1]);
    y_prob(y_out < 0)  =  y_out(y_out < 0)/abs(min([min(y_out) -1]));
    y_prob = (y_prob + 1)/2;
    
    if (~exist('acc', 'var') || isempty(acc))
        acc = (length(find(y_pred == y))/length(y));
    end
end