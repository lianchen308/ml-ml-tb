% [doubles] = cellstr2double(cells)
function [doubles] = cellstr2double(cells)
    
    doubles = cell2mat(cellfun(@str2double, cells, 'UniformOutput', false));

end