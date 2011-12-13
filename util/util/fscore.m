%fscore Calculates the fscore.
%	usage [f_score, precision, recall, accuracy] = fScore(y, y_pred)
%   y is the actual values and y_pred the predicted ones. 
%	positive labels are 1 and negative -1.
function [f_score, precision, recall, accuracy] = fscore(y, y_pred)

    true_pos  = length(find(y == y_pred & y_pred == 1));
    false_pos = length(find(y ~= y_pred & y_pred == 1));
    false_neg = length(find(y ~= y_pred & y_pred == 0));
    accuracy  = (length(find(y == y_pred))/length(y))*100;

    if (true_pos == 0)
        precision = 0;
        recall = 0;
        f_score = 0;
    else
        precision = true_pos/(true_pos + false_pos);
        recall = true_pos/(true_pos + false_neg);
        f_score = (2*precision*recall)/(precision + recall);
    end
end