function [y] = nnetBoostSim(model, X)

	y = sim(model, X');
	y = y'; 
end
