function score = giniscore(a, p)
	% Needs a and p to be column vectors.
	if (size(a,2) > 1)
		a = a';
	end
	if (size(p,2) > 1)
		p = p';
	end
	% Make sure they're the same length.
	if (length(a) != length(p))
		error ("Lengths of actual and predicted vectors differ: %d vs %d", length(a), length(p));
		return;
	end
	o = [a, p];
	k = [];
	for i = 1:length(o)
		k = [k; o(i,:), i];
	end
	k = sortrows(k, [-2, 3]);
	totalActualLosses = sum(a);
	populationDelta = 1.0/length(a);
	accumulatedPopulationPercentageSum = 0;
	accumulatedLossPercentageSum = 0;
	giniSum = 0;
	for i = 1:size(k,1)
		actual = k(i, 1);
		predicted = k(i, 2);
		accumulatedLossPercentageSum = accumulatedLossPercentageSum + (actual / totalActualLosses);
		accumulatedPopulationPercentageSum = accumulatedPopulationPercentageSum + populationDelta;
		giniSum = giniSum + accumulatedLossPercentageSum - accumulatedPopulationPercentageSum;
	end
	score = giniSum / length(a);
end