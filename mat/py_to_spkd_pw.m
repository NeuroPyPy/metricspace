function [ D ] = py_to_spkd_pw( varargin )

% Check if 'debug' argument is provided
if nargin > 0 && strcmpi(varargin{1}, 'debug')
    debugMode = true;
else
    debugMode = false;
end

% Get the directory of the script file
scriptPath = fileparts(mfilename('fullpath'));

% Get the parent directory of the script
parentPath = fullfile(scriptPath, '..');

% Add the 'STA' folder + subfolders
staFolder = fullfile(parentPath, 'STA');
addpath(staFolder);

if debugMode
    disp(['Added to path: ' staFolder]);
end

% Add subfolders of 'STA'
staSubFolders = genpath(staFolder);
addpath(staSubFolders);

if debugMode
    disp(['Added to path: ' staSubFolders]);
end

% Load the data file
filename = fullfile(scriptPath, 'spkd_vars.mat');
loadedVars = load(filename);
if ~isfield(loadedVars, 'cspks')
    error('Variable cspks does not exist in the file');
end

cspks = loadedVars.cspks;

if isstruct(cspks)
    cspks = struct2cell(cspks);
end

disp(class(cspks))

D = spkd_pw_py(cspks);
