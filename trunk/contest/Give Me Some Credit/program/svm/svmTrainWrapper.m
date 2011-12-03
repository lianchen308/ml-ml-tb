function [model] = svmTrainWrapper(X, y, C, gamma)

	options = sprintf('-q -h 0 -m 800 -b 1 -c %s -g %s', num2str ([C]), num2str ([gamma]));
	[model] = libsvmtrain(y, X, options);

end