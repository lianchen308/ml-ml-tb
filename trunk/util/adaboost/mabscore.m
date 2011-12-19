% function [acc, score] = mabscore(learners, weights, x, y, score_fcn)
function [acc, score] = mabscore(learners, weights, x, y, score_fcn)

    if (~exist('score_fcn', 'var') || isempty(score_fcn))
        score_fcn = 'aucscore';
    end

    [y_prob] = mabclassify(learners, weights, x);
    score = feval(score_fcn, y, y_prob);
	[~, ~, acc] = predlabel(y, y_prob);
end