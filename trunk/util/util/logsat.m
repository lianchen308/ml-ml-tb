%logsat Calculates the new value based on a sturation value
%	[sat_value] = logsat(value, saturation, logbase)
%   default value of logbase is saturation
%   if (value < saturation) sat_value = value
%	else sat_value = value*(log(value)/log(logbase))
function [sat_value] = logsat(value, saturation, logbase)
    if (~exist('logbase', 'var') || isempty(logbase))
        logbase = saturation;
    end
    sat_value = value;
    sat_value(value > saturation) = saturation + (log(value(value > saturation))/log(logbase));

end