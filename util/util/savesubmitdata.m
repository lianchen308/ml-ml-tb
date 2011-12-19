function savesubmitdata(file, header, y_submit_id, y_submit_prob)

    fprintf('Saving submission data...\n');
    dlmwrite(file,header,'');
    dlmwrite(file,[y_submit_id(:) y_submit_prob(:)], '-append', ...
        'delimiter',',','precision',6);
    fprintf('Data saved...\n');

end