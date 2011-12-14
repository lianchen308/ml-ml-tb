function fprintf2(fid, str, varargin)
    str = sprintf(str, varargin{:});
    fprintf(fid, str);
    fprintf(str);
end