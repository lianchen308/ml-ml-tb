% function [X_from, y_from, X_to, y_to] = movesample(X_from, y_from, X_to, y_to, X_moved)
function [X_from, y_from, X_to, y_to] = movesample(X_from, y_from, X_to, y_to, X_moved)
    
	idx = ismember(X_from, X_moved, 'rows');
    X_to = [X_to; X_from(idx, :)];
    y_to = [y_to; y_from(idx, :)];
	X_from(idx, :) = [];
    y_from(idx, :) = [];
    
end