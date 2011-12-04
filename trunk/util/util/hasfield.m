% Usage [val] = hasfield (inStruct, fieldName);
% inStruct is the name of the structure or an array of structures to search
% fieldName is the name of the field for which the function searches
function val = hasfield (struct, varargin)
   val = isfield(struct, varargin);
end