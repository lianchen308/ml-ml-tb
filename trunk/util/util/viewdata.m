function [h] = viewdata(x1, x2, y, x1_label, x2_label, fig_title)

    figure;
    h = plot(x1(y == -1), x2(y == -1), 'b.', x1(y == 1), x2(y == 1), 'r+');
    xlabel(strcat(x1_label, ' - AUC: ', num2str(aucscore(y, x1))));
    ylabel(strcat(x2_label, ' - AUC: ', num2str(aucscore(y, x2))));
    title(fig_title);
    
end

