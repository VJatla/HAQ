% 
% 
% The following script reads bounding boxes from ground truth created using
% matlab video labeler and creates a CSV file having following colmns,
%  
%     + name     = name of the video file.
%     + activity = Activity performed in the bounding box.
%     + person   = Person identity who is performing that action.
%     + W        = Width of the video.
%     + H        = Height of the video.
%     + FPS      = Frame rate of the video.
%     + T        = Video length in seconds
%     + w0       = Top left pixel location of bounding box along width.
%     + h0       = Top left pixel location of bounding box along height.
%     + f0       = Initial frame number of bounding box (POC, starts from 0).
%     + w        = width of bounding box.
%     + h        = height of bounding box.
%     + f        = Final frame number of bounding box (POC, starts form 0).,
% 
%
% Note: Pixel indexing starts from top left. That is to say top left corner
%       of video frame has pixel positon (1,1)[in case of Python (0,0)].
 


close all;
clear;
clc;

% Initializations (Change as needed)
rdir = 'C:/Users/vj/Dropbox/typing-notyping/C3L1W-D/';
min_instance_duration = 3; % in seconds

% Get all ground truth files
gt_file_keywords = ["gTruth-", ".mat"];
gt_files  = getAllFiles(rdir, gt_file_keywords); % Uses refined ground truth
% Get unique directories where gournd truth files
% are located
gt_dirs  = get_dirs(gt_files);

% Loop over each unique directory
for k = 1:length(gt_dirs)
    % Open a csv file in this directory
    file_id = fopen(gt_dirs(k)+"/gTruth.csv","w");
    fprintf(file_id,"name,activity,person,W,H,FPS,T,w0,h0,f0,w,h,f\n");
    
    % Loop over files in this directory
    gt_files = getAllFiles(char(gt_dirs(k)),gt_file_keywords);
    for i = 1:length(gt_files)

        cur_gt_fpath    = string(gt_files(i));
        [path,name,ext] = fileparts(cur_gt_fpath);

        disp(name);
        vid_name        = erase_keywords(name,gt_file_keywords) + ".mp4";
        
        vid_obj         = VideoReader(path+"/"+vid_name);
        W               = string(vid_obj.Width);
        H               = string(vid_obj.Height);
        FPS             = string(vid_obj.FrameRate);
        T               = string(round(vid_obj.Duration));
        

        gTruth          = load(cur_gt_fpath);
        gt              = timetable2table(gTruth.gTruth.LabelData);
        gt_lab_def      = gTruth.gTruth.LabelDefinitions;

        % Column loop
        colnames = gt.Properties.VariableNames(2:length(gt.Properties.VariableNames));
        for j = 1:length(colnames)

            % Create an array which has a value of 0 for empty entry
            % and 1 for the contrary.
            colname         = string(colnames(j));
            kid_name        = string(gt_lab_def.Description(find(gt_lab_def.Name == colname)));
            per_act         = kid_name.split("-");
            person          = per_act(1);
            activity        = per_act(2);
            ccol            = table2array(gt(:,colname));        
            x               = 1:length(ccol);
            func            = @(x) ~isempty(cell2mat(ccol(x)));
            ccol_not_empty  = arrayfun(func,x);

            % label continuous 1s with different labels
            [col_lab, n]    = bwlabel(ccol_not_empty);

            % go through each label and get bounding box
            for clab = 1:n
                clab_arr     = 1*(col_lab == clab);
                first_nz_pos = find(clab_arr,1,'first');
                last_nz_pos  = find(clab_arr,1,'last');
                num_frames   = last_nz_pos - first_nz_pos;
                cur_bbox     = cell2mat(table2array(gt(first_nz_pos,colname)));
                
                % Matlab indexing starts form 1 and python from 0. Here
                % to accomodate python we are subtracting 1.
                %
                %
                % The "try and catch" is to support "struct" and "array"
                % labeling methods used by Video Labeler to mark bounding
                % box.
                try
                    w0           = string(cur_bbox.Position(1)  - 1);
                    h0           = string(cur_bbox.Position(2)  - 1);
                    f0           = string(first_nz_pos - 1); 
                    w            = string(cur_bbox.Position(3));
                    h            = string(cur_bbox.Position(4));
                    f            = string(num_frames) ;
                catch
                    w0           = string(cur_bbox(1)  - 1);
                    h0           = string(cur_bbox(2)  - 1);
                    f0           = string(first_nz_pos - 1); 
                    w            = string(cur_bbox(3));
                    h            = string(cur_bbox(4));
                    f            = string(num_frames) ;
                end
                
                cur_instance_duration = floor(str2double(f))/ceil(str2double(FPS));
                
                bbox_str     = W+"," +H+"," +FPS+"," +T+"," +w0+"," +h0+"," +f0+","+...
                                w+","+h+","+f;
                csv_str      = string(vid_name) + "," + activity + "," + person;
                csv_str      = csv_str + "," + bbox_str + "\n";
                if cur_instance_duration > min_instance_duration
                    fprintf(file_id,csv_str);
                end
            end
        end
    end
    % Close csv file
    fclose(file_id);
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