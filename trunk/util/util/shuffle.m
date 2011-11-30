%shuffle Shuffles matrix along dimension dim
%	[matrix] = shuffle(matrix, dim)
function [matrix] = shuffle(matrix, dim)
    if (~exist('dim', 'var') || isempty(dim))
        dim = 1;
    end
    
    shuffle_idx = randperm(size(matrix, dim));

    if (dim == 1)
        matrix = matrix(shuffle_idx, :);
    else
        matrix = matrix(:, shuffle_idx);
    end

end