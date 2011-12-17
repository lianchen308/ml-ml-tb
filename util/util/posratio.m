% [ratio] = posratio(y)
function [ratio] = posratio(y)

    ratio = length(find(y == 1))/length(y);

end