% function [min_auc, min_acc] = mabhisteval(learners, weights, X_train, y_train, X_test, y_test) 
function [min_auc, min_acc] = mabhisteval(learners, weights, X_train, y_train, X_test, y_test) 
    min_auc = [];
	min_acc = [];
	n = length(learners);
	hist_learn = {};
	hist_weights = [];
	for i=1:length(learners)
		hist_learn{i} = learners{i};
		hist_weights(i) = weights(i);
		fprintf('Learner history %d of %d...\n', i, n);
		[min_auc, min_acc] = mabeval(hist_learn, hist_weights, X_train, y_train, X_test, y_test);
	end;
end