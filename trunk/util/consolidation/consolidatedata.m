% [variance, x1_auc, x2_auc, fig_handle] = consolidatedata(x1, x2, y, x1_label, x2_label, fig_title, plot)
function [variance, x1_auc, x2_auc] = consolidatedata(x1, x2, y, x1_label, x2_label, fig_title, plot)

    x1_auc = aucscore(y, x1);
    x2_auc = aucscore(y, x2);
    [~, ~, ~, ~, variance] = pca([x1 x2], 1);
    variance = 1 - variance(1,2);
    
    if (plot)
        figure;
        plot(x1(y == -1), x2(y == -1), 'b.', x1(y == 1), x2(y == 1), 'r+');
        xlabel([x1_label, ' - AUC: ', num2str(x1_auc)]);
        ylabel([x2_label, ' - AUC: ', num2str(x2_auc)]);
        legend('neg','pos');

        title([fig_title, ' - Var: ', num2str(variance*100, '%3.2f'), '%']);
        axis([-1 1 -1 1]);
        grid on;
    end
end

