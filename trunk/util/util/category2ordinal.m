% [categories, cat_int_val] = category2ordinal(category_cell)
function [categories, cat_int_val] = category2ordinal(category_cell, nan_val)

    if (~exist('nan_val', 'var') || isempty(nan_val))
        nan_val = 'NaN';
    end
    nan_val = lower(nan_val);

    % make the value the ordinal position
    categories = unique(lower(category_cell));
    cat_int_val = cellfun(@(val) find(ismember(categories,lower(val))), category_cell, 'UniformOutput', true);
    %cat_int_val = cell2mat(cat_int_val);
    
    % make nan_value null and remove its category
    nan_idx = find(ismember(categories,nan_val));
    if (~isempty(nan_idx))
        cat_int_val(cat_int_val == nan_idx) = NaN;
        categories{nan_idx} = {};
    end
    
end