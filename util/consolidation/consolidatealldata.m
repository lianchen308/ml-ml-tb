%[consolidated_data, data_score] = consolidatealldata(data, y, plot)
% Datastruct must contain all data to be viewed as fields, and the
% fieldnames will be used to show on figures. All fields wil be combined 2
% by 2. y is desired value to be found.
% consolidated_data will have a struct with fields x1_name, x2_name,
% variance, x1_auc and x2_auc in each row
% data_score will have all aucs by data field
function [consolidated_data, data_score] = consolidatealldata(data, y, plot)

    names = fieldnames(data); 
    n = numel(names);
    o = 1;
    for i = 1:n 
        x1_name = names{i};
        for j=i+1:n
            x2_name = names{j};
            [variance, x1_auc, x2_auc] = consolidatedata(data.(x1_name), ...
                data.(x2_name), y, x1_name, x2_name, [x1_name, ' x ', x2_name], plot);
            data_score.(x1_name) = x1_auc;
            data_score.(x2_name) = x2_auc;
            row.x1_name = x1_name;
            row.x2_name = x2_name;
            row.variance = variance;
            row.x1_auc = x1_auc;
            row.x2_auc = x2_auc;
            consolidated_data{o} = row; %#ok<AGROW>
            o = o+1;
        end
    end
    
end

