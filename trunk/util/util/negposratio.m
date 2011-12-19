% [r] = negposratio(y)
function [r] = negposratio(y)

    r = length(find(y == -1))/length(find(y == 1));

end