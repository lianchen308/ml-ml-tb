% dist = pdistmixed(x, y, is_discrete)
function dist = pdistmixed(x, y, is_discrete)
    
    if (~exist('is_discrete', 'var') || isempty(is_discrete))
        is_discrete = false(1, size(x, 2));
    end
      
    x_std = std(x);
    [y_rows ~] = size(y);
    [x_rows ~] = size(x);
    dist = zeros(x_rows, y_rows);
    for yr=1:y_rows
        % continuous distance
        x_cont = x(:, ~is_discrete);
        cont_dist = bsxfun(@minus, x_cont, y(yr, ~is_discrete));
        cont_dist = bsxfun(@rdivide, cont_dist, x_std(~is_discrete));
        
        cont_dist = sum(abs(cont_dist), 2);
        
        % discrete distance
        x_discr = x(:, is_discrete);
        discr_dist = bsxfun(@ne, x_discr, y(yr, is_discrete));
        discr_dist = sum(discr_dist, 2);
        
        dist(:, yr) = (cont_dist + discr_dist);
    end
end