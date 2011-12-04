% Returns unique row values of matrix X
%	[X, y] = uniquex(X, y)
function [X, y] = uniquex(X, y)
    
    [~, idx] = unique(X, 'rows', 'first');
	idx = sort(idx);
	X = X(idx, :);
    y = y(idx, :);

end