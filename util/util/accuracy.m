%[acc] = accuracy(y, y_pred)
function [acc] = accuracy(y, y_pred)

    acc  = (length(find(y_pred >=0 & y == 1)) + length(find(y_pred < 0 & y == -1)))/ length(y);
    
end