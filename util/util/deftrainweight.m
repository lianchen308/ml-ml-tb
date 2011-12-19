% [w] = deftrainweight(y)
function [w] = deftrainweight(y)

    w = zeros(size(y));
    w(y == -1) = 1;
    w(y == 1) = negposratio(y);

end