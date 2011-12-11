% [w] = deftrainweight(y)
function [w] = deftrainweight(y)

    w = zeros(size(y));
    w(y == -1) = 1;
    w(y == 1) = length(find(y == -1))/length(find(y == 1));

end