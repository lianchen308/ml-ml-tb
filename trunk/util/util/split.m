% [varargout] = split(x, y, varargin)
function [varargout] = split(x, y, varargin)
    
    n_start = 1;
    n = length(y);
    
    in = 1;
    for i=1:2:nargout
        n_end = min(n_start + round(varargin{in}*n), n);
        varargout{i} = x(n_start:n_end, :); %#ok<AGROW>
        varargout{i+1} = y(n_start:n_end, :); %#ok<AGROW>
        n_start = n_end + 1;
        in = in + 1;
    end
end