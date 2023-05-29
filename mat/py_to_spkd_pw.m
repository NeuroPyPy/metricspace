function [ D ] = py_to_spkd_pw()

% Get the directory of the script file
scriptPath = fileparts(mfilename('fullpath'));

% Add necessary paths relative to the script file
addpath(fullfile(scriptPath, 'for_spkd_pw')); % Add the 'for_spkd_pw' subfolder
addpath(fullfile(scriptPath, '..', '..', 'STA')); % Add the 'STA' folder
addpath(genpath(fullfile(scriptPath, '..', '..', 'STA'))); % Add subfolders of 'STA'

% Load the data file
filename = fullfile(scriptPath, 'for_spkd_pw', 'for_spkd_pw.mat');
loadedVars = load(filename);
if ~isfield(loadedVars, 'cspks')
    error('Variable cspks does not exist in the file');
end

cspks = loadedVars.cspks;

D = spkd_pw_py(cspks);
