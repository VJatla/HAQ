classdef fixed_bbox < vision.labeler.AutomationAlgorithm & vision.labeler.mixin.Temporal
    % Modified by Vj
    % PointTracker Automation algorithm to track ROIs using KLT.
    %   PointTracker is a temporal automation algorithm for tracking one or
    %   more Rectangle ROI's using KLT Feature Point Tracking in the Video
    %   Labeler and the Ground Truth Labeler Apps. Ground Truth Labeler App
    %   requires that you have the Automated Driving Toolbox(TM).
    %
    %   See also videoLabeler, vision.labeler.AutomationAlgorithm,
    %   vision.labeler.mixin.Temporal, vision.PointTracker.

    % Copyright 2017-2018 The MathWorks, Inc.
    
    %----------------------------------------------------------------------
    % Algorithm Description
    %----------------------------------------------------------------------
    properties (Constant)
        %Name Algorithm Name
        %   Character vector specifying name of algorithm.
        Name            = 'fixed_bbox';
        
        %Description Algorithm Description
        %   Character vector specifying short description of algorithm.
        Description     = 'The bouding box is fixe';
        
        %UserDirections Algorithm Usage Directions
        %   Cell array of character vectors specifying directions for
        %   algorithm users to follow in order to use algorithm.
        UserDirections  = {...
%             vision.getMessage('vision:labeler:PointTrackerROISelection'),...
%             vision.getMessage('vision:labeler:PointTrackerRun'),...
%             vision.labeler.AutomationAlgorithm.getDefaultUserDirections('review'),...
%             [vision.labeler.AutomationAlgorithm.getDefaultUserDirections('rerun') ' '...
%             vision.getMessage('vision:labeler:PointTrackerNote')],...
%             vision.labeler.AutomationAlgorithm.getDefaultUserDirections('accept')
            };
    end
    
    %----------------------------------------------------------------------
    % Tracker Properties
    %----------------------------------------------------------------------
    properties
        %InitialLabels Set of labels at algorithm initialization
        %   Table with columns Name, Type and Position for labels marked at
        %   initialization.
        InitialLabels
        
        %ImageSize Image size
        %   Size of image being processed.
        ImageSize
        
        %Trackers Cell array of trackers
        %   Cell array of point tracker objects.
        Trackers
        
        %OldPoints Cell array of points being tracked
        %   Cell array of points (feature point locations) being tracked in
        %   the same order as Trackers.
        OldPoints
        
        %IndexList Index list to InitialLabels
        %   Array mapping trackers to corresponding labels in
        %   InitialLabels.
        IndexList
        
        %BBoxPoints Bounding Box points
        %   Cell array of bounding box corner points for each tracker.
        BBoxPoints
        
        %MinimumRequiredPoints Minimum feature points to continue track
        %   Minimum number of feature points to continue tracking
        MinimumRequiredPoints = 4
    end
    
    %----------------------------------------------------------------------
    % Settings Properties
    %----------------------------------------------------------------------
    properties
        %FeatureDetectorNames List of feature detectors
        %   Cell array of character vectors containing message catalog
        %   specifier names of possible feature detectors.
        FeatureDetectorNames = {
            'MinEigenFeature'
            'HarrisFeature'
            'FASTFeature'
            'BRISKFeature'
            'KAZEFeature'
            'SURFFeature'
            'MSERFeature'};
        
        %FeatureDetectorHandles Function handles to feature detectors
        %   Function handles to feature detectors used to initialize point
        %   tracker.
        FeatureDetectorHandles = {
            @detectMinEigenFeatures
            @detectHarrisFeatures
            @detectFASTFeatures
            @detectBRISKFeatures
            @detectKAZEFeatures
            @detectSURFFeatures
            @detectMSERFeatures};
        
        %FeatureDetectorSelection Index of selected features detector
        %   Index to FeatureDetectorNames and FeatureDetectorHandles
        %   containing selected feature detector
        FeatureDetectorSelection = 1
    end
    
    %----------------------------------------------------------------------
    % Setup
    %----------------------------------------------------------------------
    methods
        function flag = supportsReverseAutomation(~)
            flag = true;
        end  
        
        function isValid = checkLabelDefinition(~, labelDef)
            
            % Only Rectangular ROI label definitions are valid for the
            % Point Tracker.
            isValid = labelDef.Type==labelType.Rectangle;
        end
        
        function isReady = checkSetup(~, videoLabels)
            
            % There must be at least one ROI label before the algorithm can
            % be executed.
            assert(~isempty(videoLabels), 'There are no ROI labels to track. Draw at least one ROI label.');
            
            isReady = true;   
        end
        
        function settingsDialog(this)
            % Create a dialog listing feature detectors to choose from.
            
            % Get translated message strings for feature detector names.
            featureDetectorStrings = cellfun(...
                @(s)vision.getMessage(sprintf('vision:labeler:%s', s)), ...
                this.FeatureDetectorNames, 'UniformOutput', false);
            
            promptString = vision.getMessage('vision:labeler:FeatureDetectorSelect');
            nameString   = sprintf('%s %s', this.Name, ...
                vision.getMessage('vision:labeler:Settings'));
            
            selection = listdlg(...
                'ListString', featureDetectorStrings,...
                'SelectionMode', 'single',...
                'Name', nameString, ...
                'PromptString', promptString,...
                'InitialValue', this.FeatureDetectorSelection, ...
                'ListSize', [250 150]);
            
            if isempty(selection)
                selection = 1;
            end
            
            this.FeatureDetectorSelection = selection;
        end
    end
    
    %----------------------------------------------------------------------
    % Execution
    %----------------------------------------------------------------------
    methods
        function initialize(algObj, I, videoLabels)
            
            % Cache initial labels marked during setup. These will be used
            % as initializations for point trackers.
            algObj.InitialLabels = videoLabels;
            
            algObj.Trackers   = {};
            algObj.OldPoints  = {};
            algObj.IndexList  = [];
            algObj.BBoxPoints = {};
            
            algObj.ImageSize  = size(I);
        end
        
        function autoLabels = run(algObj, I)
            
            autoLabels = [];
            
            % Check which labels were marked on this frame. These will be
            % used as initializations for the trackers.
            idx = algObj.InitialLabels.Time==algObj.CurrentTime;
            
            % Convert to grayscale
            Igray = rgb2gray(I);
            
            if any(idx)
                % Initialize new trackers for each of the labels marked on
                % this frame.
                idx = find(idx);
                algObj.IndexList = [algObj.IndexList; idx(:)];
                
                for n = idx'
                    initializeTrack(algObj, Igray, n);
                end
                
            else
                % Update old trackers
                numTrackers = numel(algObj.Trackers);
                for n = 1 : numTrackers
                    newLabel = updateTrack(algObj, Igray, n);
                    autoLabels = [autoLabels newLabel]; %#ok<AGROW>
                end
            end
            
        end
        
        function terminate(algObj)
            
            % Release all trackers
            for n = 1 : numel(algObj.Trackers)
                tracker = algObj.Trackers{n};
                if ~isempty(tracker)
                    release(tracker);
                end
            end
            
            % Empty arrays
            algObj.InitialLabels  = [];
            algObj.OldPoints      = {};
            algObj.IndexList      = [];
            algObj.BBoxPoints     = {};
        end
    end
    
    %----------------------------------------------------------------------
    % Private methods
    %----------------------------------------------------------------------
    methods (Access = private)
        function initializeTrack(algObj, Igray, n)
            
            % Find region of interest for feature computation
            bbox = algObj.InitialLabels{n, 'Position'};
            bboxNew = algObj.computeReducedBoundingBox(bbox);
            
            % Detect feature locations using the selected feature detector
            detectFeatures = algObj.FeatureDetectorHandles{algObj.FeatureDetectorSelection};
            points = detectFeatures( Igray, 'ROI', bboxNew );
            
            if ~isempty(points)
                % Construct a tracker
                tracker = vision.PointTracker('MaxBidirectionalError', 2);
                
                % Initialize tracker with detected features
                points = points.Location;
                tracker.initialize(points, Igray);
                bboxpoints = bbox2points(bbox);
            else
                % No features were found for this ROI. 
                tracker     = [];
                points      = [];
                bboxpoints  = [];
            end
            
            % Cache the tracker and information associated with it, to be
            % used later when updating the tracks.
            algObj.Trackers{end+1}    = tracker;
            algObj.OldPoints{end+1}   = points;
            algObj.BBoxPoints{end+1}  = bboxpoints;
        end
        
        function autoLabels = updateTrack(algObj, Igray, n)
            
            autoLabels = [];
            idx = algObj.IndexList(n);
            type = algObj.InitialLabels{idx,'Type'};
            name = algObj.InitialLabels{idx,'Name'};
            newPosition = computeROIPosition2(algObj, n);
            autoLabels = struct('Type', type, 'Name', name, 'Position', newPosition);
            
%             tracker = algObj.Trackers{n};
%             
% 			% No update needed for ROIs with no features.
%             if isempty(tracker)
%                 return;
%             end
%             
%             % Track points into new frame.
%             [points, isFound] = step(tracker, Igray);
%             
%             visiblePoints   = points(isFound,:);
%             bboxPoints      = algObj.BBoxPoints{n};
%             
%             newPosition = computeROIPosition2(algObj, bboxPoints, n);
%             oldPoints       = algObj.OldPoints{n};
%             oldInliers      = oldPoints(isFound,:);
%             
%             numPoints = size(visiblePoints, 1);
%             
%             % Some points may be lost, continue tracking only if
%             % there are enough points.
%             if numPoints >= algObj.MinimumRequiredPoints
%                 
%                 % Estimate geometric transformation between old and
%                 % new points, eliminating outliers.
%                 [tform, ~, visiblePoints, status] = ...
%                     estimateGeometricTransform(oldInliers, ...
%                     visiblePoints, 'Similarity', 'MaxDistance', 4);
%                 
%                 if status ~= 0
%                     % Break out of the tracking loop, tracking
%                     % failed.
%                     return;
%                 end
%                 
%                 % Transform the bounding box using estimated
%                 % geometric transformation
%                 bboxPoints = transformPointsForward(tform, bboxPoints);
%                 
%                 % Compute bounding box
%                 % newPosition = computeROIPosition(algObj, bboxPoints); % commented by Vj
%                 newPosition = computeROIPosition2(algObj, bboxPoints, n);
%                 
%                 % Add a new label at newPosition
%                 idx = algObj.IndexList(n);
%                 type = algObj.InitialLabels{idx,'Type'};
%                 name = algObj.InitialLabels{idx,'Name'};
%                 autoLabels = struct('Type', type, 'Name', name, 'Position', newPosition);
%                 
%                 % Update old points
%                 algObj.OldPoints{n}   = visiblePoints;
%                 algObj.BBoxPoints{n}  = bboxPoints;
%                 tracker.setPoints(visiblePoints);
%             end
        end
        
        function newBbox = computeReducedBoundingBox(~, bbox)
            
            % Use 80% of bounding box area.
            centroid = bbox(1:2) + bbox(3:4)/2;
            newExtent = 0.8 * bbox(3:4);
            
            newBbox = [centroid - newExtent/2 newExtent];
        end
        
        function position = computeROIPosition(algObj, bboxPoints)
            
            xs = bboxPoints(:,1);
            ys = bboxPoints(:,2);
            
            imHeight    = algObj.ImageSize(1);
            imWidth     = algObj.ImageSize(2);
            
            % Clamp position to image boundaries. 
            Xmin = max( min( min(xs), imWidth  ), 1 );
            Ymin = max( min( min(ys), imHeight ), 1 );
            Xmax = max( min( max(xs), imWidth  ), 1 );
            Ymax = max( min( max(ys), imHeight ), 1 );
            
            position = [Xmin Ymin Xmax-Xmin Ymax-Ymin];
        end
        
        function position = computeROIPosition2(algObj, n)
            % Changed by Vj so that the bbox never moves and changes
            % in size
            position = algObj.InitialLabels.Position(n,:);
        end
    end
end