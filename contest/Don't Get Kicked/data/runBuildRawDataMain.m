clear; clc;

tic;

fprintf('Loading training data from csv file...\n');
train_data = dataset('file','training.csv', 'delimiter',',',...
    'ReadObsNames',true, 'TreatAsEmpty', 'NULL');
n_train = length(train_data);
raw_data.ref_id_train = cellfun(@(val) str2double(val), get(train_data, 'ObsNames'), 'UniformOutput', true);

fprintf('Loading submission data from csv file...\n');
submission_data = dataset('file','test.csv', 'delimiter',',',...
    'ReadObsNames',true, 'TreatAsEmpty', 'NULL');
n_submission = length(submission_data);
raw_data.ref_id_submit = cellfun(@(val) str2double(val), get(submission_data, 'ObsNames'), 'UniformOutput', true);
submission_data.IsBadBuy = nan(n_submission,1);


% concatenating
all_data = vertcat(train_data,submission_data);
clear train_data;
clear submission_data;


fprintf('Converting Auction strings to int...\n');
[raw_data.auctions, all_data.Auction] = category2ordinal(all_data.Auction, 'NULL');
 
fprintf('Converting PurchDate strings to numbers...\n');
all_data.PurchDate  = datenum(all_data.PurchDate);
 
fprintf('Converting Make strings to int...\n');
[raw_data.makers, all_data.Make] = category2ordinal(all_data.Make, 'NULL');
 
fprintf('Converting Model strings to int...\n');
[raw_data.models, all_data.Model] = category2ordinal(all_data.Model, 'NULL');
 
fprintf('Converting Trim strings to int...\n');
[raw_data.trims, all_data.Trim] = category2ordinal(all_data.Trim, 'NULL');
 
fprintf('Converting SubModel strings to int...\n');
[raw_data.sub_models, all_data.SubModel] = category2ordinal(all_data.SubModel, 'NULL');
 
fprintf('Converting Color strings to int...\n');
[raw_data.colors, all_data.Color] = category2ordinal(all_data.Color, 'NULL');
 
fprintf('Converting Transmission strings to int...\n');
[raw_data.transmissions, all_data.Transmission] = category2ordinal(all_data.Transmission, 'NULL');
 
fprintf('Marking WheelTypeID to removal...\n');
all_data.WheelTypeID = zeros(n_train + n_submission, 1);
 
fprintf('Converting WheelType strings to int...\n');
[raw_data.wheel_types, all_data.WheelType] = category2ordinal(all_data.WheelType, 'NULL');

fprintf('Converting Nationality strings to int...\n');
[raw_data.nationalities, all_data.Nationality] = category2ordinal(all_data.Nationality, 'NULL');

fprintf('Converting Size strings to int...\n');
[raw_data.sizes, all_data.Size] = category2ordinal(all_data.Size, 'NULL');

fprintf('Converting TopThreeAmericanName strings to int...\n');
[raw_data.top_3_amer_nams, all_data.TopThreeAmericanName] = category2ordinal(all_data.TopThreeAmericanName, 'NULL');

fprintf('Converting PRIMEUNIT strings to int...\n');
[raw_data.prime_units, all_data.PRIMEUNIT] = category2ordinal(all_data.PRIMEUNIT);

fprintf('Converting AUCGUART strings to int...\n');
[raw_data.auc_guarts, all_data.AUCGUART] = category2ordinal(all_data.AUCGUART);

fprintf('Converting VNST strings to int...\n');
[raw_data.vnsts, all_data.VNST] = category2ordinal(all_data.VNST, 'NULL');

% saving rawData
all_data = double(all_data);
all_data(:, sum(abs(all_data)) == 0) = [];
raw_data.x_train = all_data(1:n_train, 2:end);
raw_data.y_train = all_data(1:n_train, 1);
raw_data.y_train(raw_data.y_train == 0) = -1;
raw_data.x_submit = all_data(n_train+1:end, 2:end);
raw_data.is_discrete = false(1, size(raw_data.x_train, 2));
raw_data.is_discrete([2 5:11 13:15 24:25 28 30]) = true;
clear all_data;

save rawData.mat raw_data;

fprintf('Raw data saved!\n\n');
toc;