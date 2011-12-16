% [categories, cat_int_val] = category2ordinal(category_cell)
function [categories, cat_int_val] = category2ordinal(category_cell)
    
    categories = unique(category_cell);
    cat_int_val = cellfun(@(val) find(ismember(categories,val)), category_cell, 'UniformOutput', false);
    cat_int_val = cell2mat(cat_int_val);

end