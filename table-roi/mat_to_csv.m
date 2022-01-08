% DESCRIPTION:
%    The following script reads annotated bounding boxes on the table and
%    creates a csv file. The CSV file is more accessable to other programs,
%    such as Python and Excel.
%
% CSV file extracted has following columns,
%   f0: 
%       POC index in session_video.mp4 (f0 = 0 implies frames 0 to 29 in
%       original video)
%   Time:
%       `session_video.mp4` time
%   <Pseudonym_1>:
%       Person 1 bounding box coordinates.
%   <Pseudonym_2>:
%       Person 2 bounding box coordinates.
%   .
%   .
%   .
%   <Pseudonym_n>:
%       Person n bounding box coordinates.

close all;
clear;
clc;


% Root directory having
%rdir = '~/Dropbox/table_roi_annotation/C1L1P-C/20170330';
rdir = 'C:\\Users\\venka\\Dropbox\\table_roi_annotation\\C1L1P-C\\20170330';     % Nimbus
rdir = 'C:\\Users\\venkatesh369\\Dropbox\\table_roi_annotation';
min_instance_duration = 3; 


% Get all ground truth files
gt_file_keywords = ["session_roi_exported.mat"];
gt_files  = getAllFiles(rdir, gt_file_keywords); % Uses refined ground truth
gt_dirs  = get_dirs(gt_files);


% Loop over each unique directory
for k = 1:length(gt_dirs)
    disp("Processing "+gt_dirs(k));

    % Load ground truth files. There should be only one file per session
    gt_files = getAllFiles(char(gt_dirs(k)),gt_file_keywords);
    if length(gt_files) > 1
       throw(MException('There should be only one ground truth file per session'))
    end
    cur_gt_fpath = string(gt_files(1));
    [path, name, ext] = fileparts(cur_gt_fpath);
    gTruth          = load(cur_gt_fpath);
    gt              = timetable2table(gTruth.gTruth.LabelData);
    
    

    % Loop over each column and change the bounding box entries to strings
    % separated by '-'.
    % For example: [0,3,4,5] becomes "0-3-4-5".
    gth = height(gt);
    gtw = width(gt);
    temp = gt;
    col_names = temp.Properties.VariableNames;
    out = array2table(string(zeros(gth, gtw)-1), 'VariableNames', col_names);

    for ridx = 1:gth
        for cidx = 1:gtw
            val = temp(ridx, cidx);
            val = table2array(val);
            if ~(iscell(val))                                   % only allows duration to pass through
                out(ridx, cidx) = table(string(val));
            else
                val = cell2mat(val);
                if length(val) > 4
                    disp(cidx, ridx);
                    throw(MException('There should be only one ground truth file per session'));
                end
                out(ridx, cidx) = table(strjoin(string(floor(val)), "-"));
            end
        end
    end

    % Creating f0 vector:
    % Each value in session_video represents 30 frames in original video.
    f0 = transpose(string(0:size(gt,1)-1));
    out.('f0') = f0;

    % Write the new table outside.
    writetable(out, path + "/session_roi.csv");
    
end 



function  output = replace_comma_with_hyphen(x)
    if isempty(x)
        output = "";
        
    else
        output = strjoin(string(x), "-");
        
    end
end

function erased_name = erase_keywords(name,gt_file_keywords)
erased_name = name;

for i=1:length(gt_file_keywords)
    cur_kw = gt_file_keywords(i);
    erased_name = erase(erased_name, cur_kw);
end

end

function fileList = getAllFiles(dirName, pattern)
% Example: filelistCSV = getAllFiles(somePath,'\d+_\d+_\d+\.csv');

  dirData = dir(dirName);      %# Get the data for the current directory
  dirIndex = [dirData.isdir];  %# Find the index for directories
  fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files
  if ~isempty(fileList)
    fileList = cellfun(@(x) fullfile(dirName,x),...  %# Prepend path to files
                       fileList,'UniformOutput',false);
    for i=1:length(pattern)
        fileList = fileList(contains(fileList, pattern(i)));
    end
    
  end
  subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
  validIndex = ~ismember(subDirs,{'.','..'});  %# Find index of subdirectories
                                               %#   that are not '.' or '..'
  for iDir = find(validIndex)                  %# Loop over valid subdirectories
    nextDir = fullfile(dirName,subDirs{iDir});    %# Get the subdirectory path
    fileList = [fileList; getAllFiles(nextDir, pattern)];  %# Recursively call getAllFiles
  end

end


function dirs  = get_dirs(files)
% Extracts unique directories where the files are located
dirs = [];
for i = 1:length(files)
    [path,name,ext] = fileparts(string(files(i)));
    dirs = [dirs;path];
end
dirs = unique(dirs);
end
