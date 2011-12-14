function [str] = struct2str(struct)
    str = struct2str_('', struct, 1);
end

function [str] = struct2str_(str, struct, level)
    fn = sort(fieldnames(struct));
    for n = 1:length(fn)
        tabs = '';
        for m = 1:level
            tabs = [tabs '    '];
        end
        str = [str tabs fn{n}]; %#ok<*AGROW>
        fn2 = struct.(fn{n});
        if isstruct(fn2)
            str = [str struct2str_('\n', fn2, level+1)];
        else
            fn2str = '';
            if (isnumeric(fn2))
                fn2str = [': ' num2str(fn2) '\n'];
            elseif (ischar(fn2))
                fn2str = [': ' fn2 '\n'];
            end
            str = [str fn2str];
        end
    end
end