% [X_resampled, y_resampled] = svmresample(X, y, c, gamma, n_batch)
function [X_resampled, y_resampled] = svmresample(X, y, c, gamma, n_batch)
    
	if (~exist('n_batch', 'var') || isempty(n_batch))
		n_batch = 10000;
	end

	X_resampled = [];
	y_resampled = [];
	n = size(y, 1);
	i = 1;
    while (i<=n)
        j = i + n_batch - 1;
        if (j > n)
            j =  n;
        end
        fprintf('Resampling %d to %d of %d...\n', i, j, n);
        X_cur = X(i:j, :);
        y_cur = y(i:j, :);
        [svm_model] = svmtrainw(X_cur, y_cur, c, gamma);
        SVs = full(svm_model.SVs);
        idx = ismember(X_cur, SVs, 'rows');
        X_resampled = [X_resampled; X_cur(idx, :)]; %#ok<AGROW> 
        y_resampled = [y_resampled; y_cur(idx, :)]; %#ok<AGROW> 
        i = j + 1;
    end
    
    [X_resampled, idx] = unique(X_resampled, 'rows');
    y_resampled = y_resampled(idx);
end
