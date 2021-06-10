%% find tracks CHECKING


%% WORKFLOW ..

clear all, close all

% detect the computer and load all_IFPEN_DARCY02_experiments
name = getenv('COMPUTERNAME');
if strcmp(name,'DESKTOP-3ONLTD9')
    cd('C:\Users\Lenovo\Jottacloud\RECHERCHE\Projets\21_IFPEN\git\Robust-Estimation')
elseif strcmp(name,'DARCY')
    cd('C:\Users\darcy\Desktop\git\Robust-Estimation')
end
load('all_IFPEN_DARCY02_experiments.mat')

iexpe = 4;

allExpeStrct(iexpe).type        = 'experiment'; % experiment / calibration
allExpeStrct(iexpe).name        = 'expe20210609_run05_200fps';
allExpeStrct(iexpe).inputFolder = ...
    strcat('E:\manipIFPEN\expe_2021_06_09\run05_200fps\');
allExpeStrct(iexpe).analysisFolder = ...
    strcat('D:\IFPEN\analysisExperiments\analysis_expe_2021_06_09\run05_200fps\');
allExpeStrct(iexpe).CalibFile = ...
    strcat('E:\manipIFPEN\expe_2021_06_09_calibration\calibrationImages\calib.mat');
allExpeStrct(iexpe).centerFinding_th = 5; % automatiser la définition de ces paramètres?
allExpeStrct(iexpe).centerFinding_sz = 2; % automatiser la définition de ces paramètres?
allExpeStrct(iexpe).maxdist = 3;          % for Benjamin tracks function:
% max distances between particules from frame to frame
allExpeStrct(iexpe).longmin = 8;         % for Benjamin tracks function:
% minimum number of points of a trajectory


iexpe = 4; % 1 / 2 / 3

allresults = struct();


allTraj = struct();

maxdist = allExpeStrct(iexpe).maxdist;
longmin = allExpeStrct(iexpe).longmin;

%%
for iSeq = iSeqa:iSeqb %35:36  % loop on images sequences
    clear trajArray_loc tracks_loc CCout
    [trajArray_loc,tracks_loc,CCout,M,filenamePlane] = ...,
        DARCY02_findTracks(allExpeStrct,iexpe,iSeq,maxdist,longmin,'figures','yes');
    allTraj(iSeq).trajArray = trajArray_loc;
    allTraj(iSeq).tracks    = tracks_loc;
    allTraj(iSeq).CC        = CCout;
    
end
fprintf('done \n')

%%


% [trajArray_CAM1,tracks_CAM1,CCout,M,filename] = ...
% DARCY02_findTracks(allExpeStrct,iexpe,ifile,maxdist,longmin,varargin)

% 1. load image
% 2. subtract mean of the image sequence
% 3. find particles positions on all images

%fprintf('line 1308 \n')
ifile = 61;
dofigures = 'yes';

fprintf('load image sequence \n')
inputFolder = allExpeStrct(iexpe).inputFolder;

cd(inputFolder)
listMcin2 = dir('*.mcin2');
filename  = listMcin2(ifile).name;
fprintf('name %s \n',listMcin2(ifile).name)

cd(inputFolder)
[~,~,params] = mCINREAD2(filename,1,1);
totalnFrames = params.total_nframes;

cd(inputFolder)
[M,~,params]=mCINREAD2(filename,1,totalnFrames);
fprintf('load image sequence - DONE \n')

% calculate mean image
ImMean = uint8(mean(M,3));
% subtract
Im01 = M - ImMean;
%%
%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%
% determine particules positions at pixel precision - then refine at
% subpixel precision
th = allExpeStrct(iexpe).centerFinding_th;
sz = allExpeStrct(iexpe).centerFinding_sz;
Nwidth = 1;
for it = 1 : size(M,3)
    %fprintf('doing time %0.4d / %0.4d \n',it,size(M,3))
    Im = zeros(size(Im01,1),size(Im01,2),class(Im01));
    Im(:,:) = Im01(:,:,it);
    
    %%%%%
    %%%%% with region props
    %%%%%
    %     stats = regionprops(Im>0,Im,'centroid','Area','weightedcentroid');
    %     ikill = find([stats.Area]<9);
    %     stats(ikill) = [];
    %
    %     for istats = 1 : length(stats)
    %         CC(it).xy(istats,1) = stats(istats).WeightedCentroid(1);
    %         CC(it).xy(istats,2) = stats(istats).WeightedCentroid(2);
    %     end
    
    %%%%%
    %%%%% with pkfnd
    %%%%%
    CC(it).xyRAW = pkfnd(Im01(:,:,it),th,sz);
    for ixy = 1 : size(CC(it).xyRAW,1)
        %refine at subpixel precision
        clear xpkfnd ypkfnd Ip
        Ip = zeros(2*Nwidth+1,2*Nwidth+1,'double');
        xpkfnd = CC(it).xyRAW(ixy,1);
        ypkfnd = CC(it).xyRAW(ixy,2);
        Ip = double(Im(ypkfnd-Nwidth:ypkfnd+Nwidth,xpkfnd-Nwidth:xpkfnd+Nwidth));
        CC(it).xy(ixy,1) = xpkfnd + 0.5*log(Ip(2,3)/Ip(2,1))/(log((Ip(2,2)*Ip(2,2))/(Ip(2,1)*Ip(2,3))));
        CC(it).xy(ixy,2) = ypkfnd + 0.5*log(Ip(3,2)/Ip(1,2))/(log((Ip(2,2)*Ip(2,2))/(Ip(1,2)*Ip(3,2))));
        
    end
end
%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%
% put all positions in only one variable called CCall
% also in CCout, an output of the function for later calculations
clear CCall %= [];
for it = 1 : size(M,3)
    X = CC(it).xy(:,1);
    Y = CC(it).xy(:,2);
    CCout(it).X = X;
    CCout(it).Y = Y;
    T = it * ones(1,length(X));
    if it == 1
        CCall = [X,Y];
        CCall(:,3) = [T];
    else
        CCtemp = [X,Y];
        CCtemp(:,3) = [T];
        CCall = [CCall;CCtemp];
    end
end


%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%
% find tracks and stitch them
% prepare data for TAN_track2d
clear trajArray_CAM1 tracks_CAM1 part_cam1
for it = 1 : size(M,3)
    idxt = find(CCall(:,3)==it);
    part_cam1(it).pos(:,1) = [CCall(idxt,1)]; % out_CAM1(:,1);
    part_cam1(it).pos(:,2) = [CCall(idxt,2)]; % out_CAM1(:,2);
    part_cam1(it).pos(:,3) = ones(length([CCall(idxt,1)]),1)*it;
    part_cam1(it).intensity = 0; %mI;
end

% maxdist = 3;
% longmin = 5;
[trajArray_CAM1,tracks_CAM1] = TAN_track2d(part_cam1,maxdist,longmin);
% coluns of trajArray_CAM1 length(trajArray_CAM1) is n° of trajectories
% column 1: X
% column 2: Y
% column 3: t
% column 4: n° trajectory
% column 5: state of particle: 0: free 1: not free  2: linked to two or
% more other particles
%
% coluns of tracks_CAM1 length(tracks_CAM1) is n° of frames
% column 1: X
% column 2: Y
% column 3: t
% column 4: n° trajectory
% column 5: state of particle: 0: free 1: not free  2: linked to two or
% more other particles

%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%
% only doing figure here
switch dofigures
    case 'yes'
        
        figure
        imagesc(ImMean)
        figure
        imhist(Im01)
        
        figure
        imagesc(Im01(:,:,1))
        hold on
        plot(CC(1).xy(:,1),CC(1).xy(:,2),'ob')
        
        tmin = 0001;
        tmax = size(M,3);
        colP = parula(tmax-tmin+1);
        ht = figure('defaultAxesFontSize',20); hold on, box on
        set(gca,'ydir','reverse')
        set(gcf,'position', [474    98   948   866])
        axis([0 1152 0 1152])
        h = patch('Faces',[1:4],'Vertices',[0 0;1152 0;1152 1152;0 1152]);
        h.FaceColor = [.1 .1 .1];
        h.EdgeColor = 'none';
        h.FaceAlpha = .8;
        for it = tmin : tmax
            idxt = find(CCall(:,3)==it);
            hp = plot(CCall(idxt,1),CCall(idxt,2),'ok',...
                'MarkerEdgeColor','none','markerFaceColor',colP(it,:));
            %pause(.1)
        end
        
        clear Xtck Ytck tckSize
        Xtck = []; Ytck = []; tckSize = [];
        figure(ht), hold on
        for it = 1 : length(trajArray_CAM1)
            tckSize(it) = length(trajArray_CAM1(it).track(:,1));
        end
        Xtck = NaN(length(trajArray_CAM1),max(tckSize));
        Ytck = NaN(length(trajArray_CAM1),max(tckSize));
        
        for it = 1 : length(trajArray_CAM1)
            Xtck(it,1:length(trajArray_CAM1(it).track(:,1))) = ...
                trajArray_CAM1(it).track(:,1);
            Ytck(it,1:length(trajArray_CAM1(it).track(:,1))) = ...
                trajArray_CAM1(it).track(:,2);
        end
        % htrck = plot(Xtck',Ytck','-','lineWidth',4);
end
%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%

%%
clear Xtck Ytck tckSize
Xtck = []; Ytck = []; tckSize = [];
figure('defaultAxesFontSize',20), hold on
for it = 1 : length(trajArray_CAM1)
    tckSize(it) = length(trajArray_CAM1(it).track(:,1));
end
Xtck = NaN(length(trajArray_CAM1),max(tckSize));
Ytck = NaN(length(trajArray_CAM1),max(tckSize));

for it = 1 : length(trajArray_CAM1)
    Xtck(it,1:length(trajArray_CAM1(it).track(:,1))) = ...
        trajArray_CAM1(it).track(:,1);
    Ytck(it,1:length(trajArray_CAM1(it).track(:,1))) = ...
        trajArray_CAM1(it).track(:,2);
end
 htrck = plot(Xtck',Ytck','-','lineWidth',4);
 set(gca,'ydir','reverse')
 
%% tests on stitching
% show the longest track
[~,itb] = max(tckSize);
figure('defaultAxesFontSize',20), box on, hold on
htrck = plot(Xtck',Ytck','-','Color',0.5*[1 1 1],'lineWidth',4);
%%
plot(trajArray_CAM1(itb).track(:,1),trajArray_CAM1(itb).track(:,2),'color','b','lineWidth',4)
%% Working on the stitching 

timeShift = 5; % frames

% stitch before
pA = [43.4315,729.788];
pB = [43.3156,729.711];
hold on
plot(pA(1),pA(2),'or')
plot(pB(1),pB(2),'og')
% find tracks from pA and pB
clear dA dB itA itB
for it = 1 : length(trajArray_CAM1)
    clear dd xt yt
    xt = trajArray_CAM1(it).track(:,1);
    yt = trajArray_CAM1(it).track(:,2);
    dd = sqrt((xt-pA(1)).^2+(yt-pA(2)).^2);
    dA(it) = min(dd);
end
for it = 1 : length(trajArray_CAM1)
    clear dd xt yt
    xt = trajArray_CAM1(it).track(:,1);
    yt = trajArray_CAM1(it).track(:,2);
    dd = sqrt((xt-pB(1)).^2+(yt-pB(2)).^2);
    dB(it) = min(dd);
end
[~,itA] = min(dA);
[~,itB] = min(dB);
plot(trajArray_CAM1(itB).track(:,1),trajArray_CAM1(itB).track(:,2),'color','r','lineWidth',4)

tminA = min(trajArray_CAM1(itA).track(:,3));
tmaxA = max(trajArray_CAM1(itA).track(:,3));
tminB = min(trajArray_CAM1(itB).track(:,3));
tmaxB = max(trajArray_CAM1(itB).track(:,3));
%%

% I indicate a trajectory - itA 
% it selects all trajectories that end timeShift before the start of my
% choosen trajectory
tminA = min(trajArray_CAM1(itA).track(:,3));
itBcandidates = [];
for itB = 1 : length(trajArray_CAM1)
    tmaxB = max(trajArray_CAM1(itB).track(:,3));
    if (tminA - tmaxB) < timeShift && (tminA - tmaxB) >0
        itBcandidates = [itBcandidates,itB];
        fprintf('itB: %0.0f,tminA: %0.0f,tmaxB: %0.0f \n',itB,tminA,tmaxB)
        hh(itB) = plot(trajArray_CAM1(itB).track(:,1),trajArray_CAM1(itB).track(:,2),'color','r','lineWidth',4);
    end
end
%%

trajAx = smoothdata(trajArray_CAM1(itA).track(:,1),'gaussian',20);
trajAy = smoothdata(trajArray_CAM1(itA).track(:,2),'gaussian',20);
figure, hold on
plot(trajArray_CAM1(itA).track(:,1),trajArray_CAM1(itA).track(:,2),'r-o')
plot(trajAx,trajAy,'b-o')

trajBx = smoothdata(trajArray_CAM1(itB).track(:,1),'gaussian',20);
trajBy = smoothdata(trajArray_CAM1(itB).track(:,2),'gaussian',20);
plot(trajArray_CAM1(itB).track(:,1),trajArray_CAM1(itB).track(:,2),'r-o')
plot(trajBx,trajBy,'b-o')

%%
trajectories = randn(100,7);
% Taking mean across the trajectories (that is 2nd dimension) will give a single trajectory
trajectory = mean(trajectories,2);
p = 0.1; % define the error measure weightage (Give path matching closely with datapoint for high value of p)
% p must be between 0 and 1.
x_data = linspace(1,100,100); % Just to have 2D sense of trajectories
plot(trajectory(:,1),trajectory(:,2))
%%


% Lets assume we have 7 trajectories with 100 datapoints each. 
% For example datapoints can be generated using randn
trajectories = randn(100,7);
% Taking mean across the trajectories (that is 2nd dimension) will give a single trajectory
trajectory = mean(trajectories,2);
p = 0.1; % define the error measure weightage (Give path matching closely with datapoint for high value of p)
% p must be between 0 and 1.
x_data = linspace(1,100,100); % Just to have 2D sense of trajectories
path = csaps(x_data,trajectory,p);
fnplt(path); % show the path
% Here path is a structure which contains the polynomial coefficient between each successive pair of datapoint.

