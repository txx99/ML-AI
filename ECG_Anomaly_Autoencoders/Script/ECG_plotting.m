% plotting ECG data
clc;
clearvars;
close all;
% read in table, convert to array for downstream
    % ecg_zip = './archive.zip'
    % unzip(ecg_path)
ecg_path = '../Data/ecg.csv';
ecg_array = table2array(readtable(ecg_path));

% visualise; only 1 col  
plot(ecg_array(:, 1))
% lets superimpose data from all cols 1:140 
% last col is 0 1 target col, if ECG data is normal or abnormal
for i = 1:(length(ecg_array)-1)
    plot(ecg_array(:, i));
    hold on % to not overwrite prior graph
end