function score = giniscore(y, y_pred)
%--------------------------------------------------------------------------
%
% This function calculates the non-normalized gini index. To normalize,
% divide result by ginicalc(y,y).
%
% gini=giniscore(y,y_pred)
%
% y = an [n,1] array of actual probabilities
%
% y_pred = an [n,1] array of predicted probabilities
%
% gini is the non-normalized gini index. Higher values indicate
% that the probabilities in y_pred are a better fit to y.
%
%--------------------------------------------------------------------------
	y=y(:);
	y_pred=y_pred(:);
    y(y == -1) = 0;
	n=length(y);
	k=[y,y_pred,(1:n)'];
	k=sortrows(k,[-2,3]);
	score=sum(cumsum(k(:,1)./sum(k(:,1)))-(1/n:1/n:1)')/n;
end