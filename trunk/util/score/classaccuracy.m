%[negacc, posacc] = classaccuracy(y, y_pred)
function [negacc, posacc] = classaccuracy(y, y_pred)

    posacc  = length(find(y_pred >=0 & y == 1))/length(find(y == 1));
    negacc  = length(find(y_pred < 0 & y == -1))/length(find(y == -1));
    
end