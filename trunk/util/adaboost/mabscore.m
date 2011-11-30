% function [y_acc, y_auc] = mabscore(learners, weights, X, y)
function [acc, auc] = mabscore(learners, weights, X, y)
    [y_prob] = mabclassify(learners, weights, X);
    auc = aucscore(y, y_prob);
	[~, ~, acc] = predlabel(y, y_prob);
end