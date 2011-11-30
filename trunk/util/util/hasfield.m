% Usage [val] = hasfield (inStruct, fieldName);
% inStruct is the name of the structure or an array of structures to search
% fieldName is the name of the field for which the function searches
function val = hasfield (struct, field_name)
    val = 0;
    f = fieldnames(struct(1));
    for i=1:length(f)
        if(strcmp(f{i},strtrim(field_name)))
            val = 1;
            return;
        elseif isstruct(struct(1).(f{i}))
            val = hasfield(struct(1).(f{i}), field_name);
            if val
                return;
            end
        end
    end
end