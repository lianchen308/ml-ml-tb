%returns unique values shuffled
%	[X, y] = uniquex(X, y)
function [X, y] = uniquex(X, y)
    
    [X, idx] = unique(X, 'rows');
    y = y(idx);
    
    [X, idx] = shuffle(X);
    y = y(idx);

end