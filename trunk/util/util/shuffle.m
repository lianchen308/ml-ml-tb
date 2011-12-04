%shuffle Shuffles matrix along dimension dim
%	[matrix, idx] = shuffle(matrix, dim)
function [matrix, idx] = shuffle(matrix, dim)
    if (~exist('dim', 'var') || isempty(dim))
        dim = 1;
    end
    
    idx = randperm(size(matrix, dim));

    if (dim == 1)
        matrix = matrix(idx, :);
    else
        matrix = matrix(:, idx);
    end

end