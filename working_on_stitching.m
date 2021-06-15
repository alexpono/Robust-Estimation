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

iexpe = 5;

allExpeStrct(iexpe).type        = 'experiment'; % experiment / calibration
allExpeStrct(iexpe).name        = 'expe_2021_06_14_run05_statistics';
allExpeStrct(iexpe).inputFolder = ...
    strcat('E:\manipIFPEN\expe_2021_06_14\run05_statistics\');
allExpeStrct(iexpe).analysisFolder = ...
    strcat('D:\IFPEN\analysisExperiments\analysis_expe_2021_06_14\run05_statistics\');
allExpeStrct(iexpe).CalibFile = ...
    strcat('E:\manipIFPEN\expe_2021_06_09_calibration\calibrationImages\calib.mat');
allExpeStrct(iexpe).centerFinding_th = 2; % automatiser la définition de ces paramètres?
allExpeStrct(iexpe).centerFinding_sz = 1; % automatiser la définition de ces paramètres?
allExpeStrct(iexpe).maxdist = 3;          % for Benjamin tracks function:
% max distances between particules from frame to frame
allExpeStrct(iexpe).longmin = 8;         % for Benjamin tracks function:
% minimum number of points of a trajectory

allresults = struct();


allTraj = struct();

maxdist = allExpeStrct(iexpe).maxdist;
longmin = allExpeStrct(iexpe).longmin;

%%
iSeqa= 1 
iSeqb= 2
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
ifile = 1;
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
%%
%%%% %%%% %%%%
%%%% %%%% %%%% calculate ds
%%%% %%%% %%%%
for itraj = 1 : size(trajArray_CAM1,2)
    for ittime = 2 : size(trajArray_CAM1(itraj).track,1)
        xi = trajArray_CAM1(itraj).track(ittime-1,1);
        yi = trajArray_CAM1(itraj).track(ittime-1,2);
        xf = trajArray_CAM1(itraj).track(ittime,1);
        yf = trajArray_CAM1(itraj).track(ittime,2);
        trajArray_CAM1(itraj).track(ittime,6) = sqrt((xf-xi)^2+(yf-yi)^2);
    end
    % subset the trajectory with a point every 5 steps in time
    iit = 0;
    % prepare timeSubSample
    itStep = 4;
    iti = 1+itStep/2;
    itf = size(trajArray_CAM1(itraj).track,1)-itStep/2;
    timeSubSample_i = iti :  itStep : round(itf/2);
    timeSubSample_f = itf : -itStep : round(itf/2);
    if timeSubSample_i(end) == timeSubSample_f(end)
        timeSubSample_f(end) = [];
        timeSubSample = [timeSubSample_i,flip(timeSubSample_f)];
    elseif (timeSubSample_f(end)-timeSubSample_i(end)) > 5
        timeSubSample_i = [timeSubSample_i,round((timeSubSample_i(end)+timeSubSample_f(end))/2)];
        timeSubSample = [timeSubSample_i,flip(timeSubSample_f)];
    else
        timeSubSample = [timeSubSample_i,flip(timeSubSample_f)];
    end
    
    for iittime = 1 : length(timeSubSample) % 1 : 5 : size(trajArray_CAM1(itraj).track,1)
        iit = iit + 1;
        ittime = timeSubSample(iittime);
        tmean_i = ittime-itStep/2;
        tmean_f = ittime+itStep/2;
%         if ittime == 1
%         tmean_i = ittime;
%         tmean_f = ittime+3;
%         elseif ittime == timeSubSample(end)
%         tmean_i = ittime-3;
%         tmean_f = ittime;
%         else
%         tmean_i = ittime;
%         tmean_f = min(ittime+5,timeSubSample(end));
%         end
        trajArray_CAM1(itraj).smplTrack(iit,1) = ...
            mean( trajArray_CAM1(itraj).track(tmean_i:tmean_f,1));
        trajArray_CAM1(itraj).smplTrack(iit,2) = ...
            mean( trajArray_CAM1(itraj).track(tmean_i:tmean_f,2)); 
        trajArray_CAM1(itraj).smplTrack(iit,3) = trajArray_CAM1(itraj).track(ittime,3); 
        trajArray_CAM1(itraj).smplTrack(iit,4) = trajArray_CAM1(itraj).track(ittime,4); 
        trajArray_CAM1(itraj).smplTrack(iit,5) = trajArray_CAM1(itraj).track(ittime,5); 
    end
    
    for ittime = 2 : size(trajArray_CAM1(itraj).smplTrack,1)
        xi = trajArray_CAM1(itraj).smplTrack(ittime-1,1);
        yi = trajArray_CAM1(itraj).smplTrack(ittime-1,2);
        xf = trajArray_CAM1(itraj).smplTrack(ittime,1);
        yf = trajArray_CAM1(itraj).smplTrack(ittime,2);
        trajArray_CAM1(itraj).smplTrack(ittime,6) = sqrt((xf-xi)^2+(yf-yi)^2);
    end
    
    trajArray_CAM1(itraj).dsSUM    = sum([trajArray_CAM1(itraj).track(:,6)]);
    trajArray_CAM1(itraj).dsSUMwindow = sum([trajArray_CAM1(itraj).smplTrack(:,6)]);
end
%%%% %%%% %%%%
%%%% %%%% %%%% calculate ds END
%%%% %%%% %%%%


%%
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

%% showing trajectories 
colTraj = jet(201);
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

hold on, plot(CCall(:,1),CCall(:,2),'ok')
htrck = plot(Xtck',Ytck','-','lineWidth',4);
set(gca,'ydir','reverse')

%% showing trajectories  V2
colTraj = jet(201);
clear Xtck Ytck tckSize
Xtck = []; Ytck = []; tckSize = [];
figure('defaultAxesFontSize',20), hold on
for it = 1 : length(trajArray_CAM1)
    tckSize(it) = length(trajArray_CAM1(it).track(:,1));
end
Xtck = NaN(length(trajArray_CAM1),max(tckSize));
Ytck = NaN(length(trajArray_CAM1),max(tckSize));

hold on, 
for ip = 1 : length(CCall)
    plot(CCall(ip,1),CCall(ip,2),'ok','markerFaceColor',colTraj(CCall(ip,3),:))
end

for it = 1 : length(trajArray_CAM1)
    Xtck(it,1:length(trajArray_CAM1(it).track(:,1))) = ...
        trajArray_CAM1(it).track(:,1);
    Ytck(it,1:length(trajArray_CAM1(it).track(:,1))) = ...
        trajArray_CAM1(it).track(:,2); 
    colT = colTraj(trajArray_CAM1(it).track(1,3),:);
    plot(Xtck(it,:),Ytck(it,:),'-','lineWidth',4,'color',colT)
end


%htrck = plot(Xtck',Ytck','-','lineWidth',4);
set(gca,'ydir','reverse')

   
%% I indicate a point, it finds the trajectory
pX = 92.0974; pY =  959.297;
pX = 95.3814; pY =  954.207;

clear dP
for itrj = 1 : size(trajArray_CAM1,2)
    clear dd
    for itrck = 1 : size(trajArray_CAM1(itrj).track,1)
        xl = trajArray_CAM1(itrj).track(itrck,1);
        yl = trajArray_CAM1(itrj).track(itrck,2);
        dd(itrck) = sqrt((pX-xl)^2+(pY-yl)^2);
    end
    dP(itrj) = min(dd);
end
[~,itrj] = min(dP)
%% tests on stitching
% show the longest track
[~,itb] = max(tckSize);
[~,itb] = max([trajArray_CAM1.dsSUMwindow]);
itb = 98;%149
%[~,itb] = maxk([trajArray_CAM1.dsSUMwindow],10);
figure('defaultAxesFontSize',20), box on, hold on
htrck = plot(Xtck',Ytck','-','Color',0.5*[1 1 1],'lineWidth',4);
for ids = 1 : length(itb)
    plot(trajArray_CAM1(itb(ids)).track(:,1),trajArray_CAM1(itb(ids)).track(:,2),'color','r','lineWidth',4)
end
%% Working on the stitching
figure('defaultAxesFontSize',20), box on, hold on
htrck = plot(Xtck',Ytck','-','Color',0.5*[1 1 1],'lineWidth',4);

%%
% trajArray_CAM1_BACKUP = trajArray_CAM1;
trajArray_CAM1 = trajArray_CAM1_BACKUP;
%%
lcrossStitchTHRSHLD = 4;

itList = [1 : size(trajArray_CAM1(itA).track  ,1)];
continue2stich = 'on'
conversionsSTR = struct();
icSTR = 0;
while strcmp(continue2stich,'on') % as long as we can stitch we continue to stitch
    itA = 0;
    conversions = 0 ;
    while(1)% trouver une trach qui peut stitcher
        itA = itA+1;
        if itA > size(trajArray_CAM1,2)
        fprintf('stitched %0.0f trajectories \n',conversions)
            if conversions == 0
                continue2stich = 'off'
            end
            break
        end
    %pause(.1)
    %itA = 98%149;
    if exist('hh'),     delete(hh),     end
    if exist('hitA'),   delete(hitA),   end
    if exist('hhSMPL'), delete(hhSMPL), end
    if exist('hitASMPL'), delete(hitASMPL), end
    if exist('hA'), delete(hA), end
    if exist('hB'), delete(hB), end
    if exist('hcA'), delete(hcA), end
    if exist('hAextra'), delete(hAextra), end
    if exist('hcB'), delete(hcB), end
    if exist('hBextra'), delete(hBextra), end
    
    timeShift = 5; % 5; % frames
    
    hitA     = plot(trajArray_CAM1(itA).track(:,1),...
        trajArray_CAM1(itA).track(:,2),'.','color','b','lineWidth',4);
    hitASMPL = plot(trajArray_CAM1(itA).smplTrack(:,1),...
        trajArray_CAM1(itA).smplTrack(:,2),'ob');
    % I indicate a trajectory - itA
    % it selects all trajectories that end timeShift before the start of my
    % choosen trajectory
    % tminA = min(trajArray_CAM1(itA).track(:,3));
    % itBcandidates = [];
    % for itB = 1 : length(trajArray_CAM1)
    %     tmaxB = max(trajArray_CAM1(itB).track(:,3));
    %     if (tminA - tmaxB) < timeShift && (tminA - tmaxB) >0
    %         itBcandidates = [itBcandidates,itB];
    %         fprintf('itB: %0.0f,tminA: %0.0f,tmaxB: %0.0f \n',itB,tminA,tmaxB)
    %         hh(itB) = plot(trajArray_CAM1(itB).track(:,1),trajArray_CAM1(itB).track(:,2),'color','r','lineWidth',4);
    %     end
    % end
    
    rA = 20;
    tmaxA = max(trajArray_CAM1(itA).track(:,3));
    itBcandidates = [];
    dABall = [];
    dCrossingCandidates = [];
    for itB = 1 : length(trajArray_CAM1)
        tminB = min(trajArray_CAM1(itB).track(:,3));
        if (tminB - tmaxA) < timeShift && (tminB - tmaxA) > 0
            clear xA yA xB yB dAB
            xA = trajArray_CAM1(itA).track(end,1);
            yA = trajArray_CAM1(itA).track(end,2);
            xB = trajArray_CAM1(itB).track(1,1);
            yB = trajArray_CAM1(itB).track(1,2);
            dAB = sqrt((xA-xB)^2+(yA-yB)^2);
            if dAB < rA
                dABall = [dABall,dAB];
                itBcandidates = [itBcandidates,itB];
                fprintf('itA: %0.0f,  itB: %0.0f, dAB: %0.0f,tmaxA: %0.0f,tminB: %0.0f \n',itA,itB,dAB,tmaxA,tminB)
                hh(itB) = plot(trajArray_CAM1(itB).track(:,1),...
                    trajArray_CAM1(itB).track(:,2),'>r','color','r','lineWidth',4);
                hhSMPL(itB) = plot(trajArray_CAM1(itB).smplTrack(:,1),...
                    trajArray_CAM1(itB).smplTrack(:,2),'or');
                hA = plot(xA,yA,'>k');
                hB = plot(xB,yB,'sk');
                % extrapolate the position of traj A and B and show where the tracer would be
                tA2B = (tmaxA+tminB)/2;
                Dt = (trajArray_CAM1(itA).smplTrack(end,3)-trajArray_CAM1(itA).smplTrack(end-1,3));
                Dx = (trajArray_CAM1(itA).smplTrack(end,1)-trajArray_CAM1(itA).smplTrack(end-1,1));
                Dy = (trajArray_CAM1(itA).smplTrack(end,2)-trajArray_CAM1(itA).smplTrack(end-1,2));
                vAsmplX = Dx / Dt;
                vAsmplY = Dy / Dt;
                xA_extra = trajArray_CAM1(itA).smplTrack(end,1) + ...
                    (tA2B - trajArray_CAM1(itA).smplTrack(end,3)) * vAsmplX;
                yA_extra = trajArray_CAM1(itA).smplTrack(end,2) + ...
                    (tA2B - trajArray_CAM1(itA).smplTrack(end,3)) * vAsmplY;
                hcA = viscircles([xA_extra,yA_extra]  , 2,'Color','b');
                hAextra = plot([trajArray_CAM1(itA).smplTrack(end,1),xA_extra],...
                    [trajArray_CAM1(itA).smplTrack(end,2),yA_extra] , '--b');
                
                Dt = (trajArray_CAM1(itB).smplTrack(2,3)-trajArray_CAM1(itB).smplTrack(1,3));
                Dx = (trajArray_CAM1(itB).smplTrack(1,1)-trajArray_CAM1(itB).smplTrack(2,1));
                Dy = (trajArray_CAM1(itB).smplTrack(1,2)-trajArray_CAM1(itB).smplTrack(2,2));
                
                vBsmplX = Dx/Dt;
                vBsmplY = Dy/Dt;
                xB_extra = trajArray_CAM1(itB).smplTrack(1,1) + ...
                    (- tA2B + trajArray_CAM1(itB).smplTrack(1,3)) * vBsmplX;
                yB_extra = trajArray_CAM1(itB).smplTrack(1,2) + ...
                    (- tA2B + trajArray_CAM1(itB).smplTrack(1,3)) * vBsmplY;
                hcB = viscircles([xB_extra,yB_extra]  , 2,'Color','r');
                hBextra = plot([trajArray_CAM1(itB).smplTrack(1,1),xB_extra],...
                    [trajArray_CAM1(itB).smplTrack(1,2),yB_extra] , '--r');
                
                dCrossingCandidates = [dCrossingCandidates,sqrt((xB_extra-xA_extra)^2+(yB_extra-yA_extra)^2)];
            end
        end
    end
    
    
    % stitch the best candidate if it is possible
    [mindist,itBstitch] = min(dCrossingCandidates);
    if mindist < lcrossStitchTHRSHLD        
        itB = itBcandidates(itBstitch);
        
        conversions = conversions + 1;
        
        icSTR = icSTR + 1;
        conversionsSTR(icSTR).mindist = mindist;
        conversionsSTR(icSTR).tmaxA = tmaxA; 
        conversionsSTR(icSTR).tminB = trajArray_CAM1(itB).track(1,3); 
        conversionsSTR(icSTR).itA = itA;
        conversionsSTR(icSTR).itB = trajArray_CAM1(itB).track(1,3); 
        conversionsSTR(icSTR).Ax = trajArray_CAM1(itA).track(:,1);
        conversionsSTR(icSTR).Ay = trajArray_CAM1(itA).track(:,2);
        conversionsSTR(icSTR).At = trajArray_CAM1(itA).track(:,3);
        conversionsSTR(icSTR).Bx = trajArray_CAM1(itB).track(:,1);
        conversionsSTR(icSTR).By = trajArray_CAM1(itB).track(:,2);
        conversionsSTR(icSTR).Bt = trajArray_CAM1(itB).track(:,3);
        xA = trajArray_CAM1(itA).track(end,1);
        yA = trajArray_CAM1(itA).track(end,2);
        xB = trajArray_CAM1(itB).track(1,1);
        yB = trajArray_CAM1(itB).track(1,2);
        dAB = sqrt((xA-xB)^2+(yA-yB)^2);
        conversionsSTR(icSTR).dAB = dAB;
        conversionsSTR(icSTR).xA_extra = xA_extra;
        conversionsSTR(icSTR).yA_extra = yA_extra;
        conversionsSTR(icSTR).xB_extra = xB_extra;
        conversionsSTR(icSTR).yB_extra = yB_extra;
        
        % attach B to A
        xA_f = trajArray_CAM1(itA).track(end,1);
        yA_f = trajArray_CAM1(itA).track(end,2);
        xB_i = trajArray_CAM1(itB).track(1,1);
        yB_i = trajArray_CAM1(itB).track(1,2);
        tA_f = trajArray_CAM1(itA).track(end,3);
        tB_i = trajArray_CAM1(itB).track(1,3);
        for it = tA_f+1 : tB_i-1
            trajArray_CAM1(itA).track(end+1,1) =  xA_f + (xB_i - xA_f) * ((it-tA_f)/(tB_i-tA_f)) ;
            trajArray_CAM1(itA).track(end,2)   =  yA_f + (yB_i - yA_f) * ((it-tA_f)/(tB_i-tA_f)) ;
            trajArray_CAM1(itA).track(end,3)   =  it;
            trajArray_CAM1(itA).track(end,4)   =  0;
        end
        LitA = size(trajArray_CAM1(itA).track  ,1);
        for it = 1 : size(trajArray_CAM1(itB).track  ,1)
            trajArray_CAM1(itA).track(LitA+it,1) =  trajArray_CAM1(itB).track(it,1);
            trajArray_CAM1(itA).track(LitA+it,2) =  trajArray_CAM1(itB).track(it,2);
            trajArray_CAM1(itA).track(LitA+it,3) =  trajArray_CAM1(itB).track(it,3);
            trajArray_CAM1(itA).track(LitA+it,4) =  trajArray_CAM1(itB).track(it,4);
        end
        
        % kill B
        trajArray_CAM1(itB) = [];
        
        % recalculate smplTrack
        trajArray_CAM1(itA).smplTrack  = [];
        itraj = itA;
        iti = 1+itStep/2;
        itf = size(trajArray_CAM1(itraj).track,1)-itStep/2;
        timeSubSample_i = iti :  itStep : round(itf/2);
        timeSubSample_f = itf : -itStep : round(itf/2);
        if timeSubSample_i(end) == timeSubSample_f(end)
            timeSubSample_f(end) = [];
            timeSubSample = [timeSubSample_i,flip(timeSubSample_f)];
        elseif (timeSubSample_f(end)-timeSubSample_i(end)) > 5
            timeSubSample_i = [timeSubSample_i,round((timeSubSample_i(end)+timeSubSample_f(end))/2)];
            timeSubSample = [timeSubSample_i,flip(timeSubSample_f)];
        else
            timeSubSample = [timeSubSample_i,flip(timeSubSample_f)];
        end
        
        iit = 0;
        for iittime = 1 : length(timeSubSample) % 1 : 5 : size(trajArray_CAM1(itraj).track,1)
            iit = iit + 1;
            ittime = timeSubSample(iittime);
            tmean_i = ittime-itStep/2;
            tmean_f = ittime+itStep/2;
            trajArray_CAM1(itraj).smplTrack(iit,1) = ...
                mean( trajArray_CAM1(itraj).track(tmean_i:tmean_f,1));
            trajArray_CAM1(itraj).smplTrack(iit,2) = ...
                mean( trajArray_CAM1(itraj).track(tmean_i:tmean_f,2));
            trajArray_CAM1(itraj).smplTrack(iit,3) = trajArray_CAM1(itraj).track(ittime,3);
            trajArray_CAM1(itraj).smplTrack(iit,4) = trajArray_CAM1(itraj).track(ittime,4);
            trajArray_CAM1(itraj).smplTrack(iit,5) = trajArray_CAM1(itraj).track(ittime,5);
        end
        
        for ittime = 2 : size(trajArray_CAM1(itraj).smplTrack,1)
            xi = trajArray_CAM1(itraj).smplTrack(ittime-1,1);
            yi = trajArray_CAM1(itraj).smplTrack(ittime-1,2);
            xf = trajArray_CAM1(itraj).smplTrack(ittime,1);
            yf = trajArray_CAM1(itraj).smplTrack(ittime,2);
            trajArray_CAM1(itraj).smplTrack(ittime,6) = sqrt((xf-xi)^2+(yf-yi)^2);
        end
        
        trajArray_CAM1(itraj).dsSUM    = sum([trajArray_CAM1(itraj).track(:,6)]);
        trajArray_CAM1(itraj).dsSUMwindow = sum([trajArray_CAM1(itraj).smplTrack(:,6)]);
    end
    
    end
end
%% showing each stitched trajectory
icSTR = 28;
figure('defaultAxesFontSize',20), hold on, box on
plot(conversionsSTR(icSTR).Ax,conversionsSTR(icSTR).Ay,'ob')
plot(conversionsSTR(icSTR).Bx,conversionsSTR(icSTR).By,'or')
plot(conversionsSTR(icSTR).xA_extra,conversionsSTR(icSTR).yA_extra,'ob','markerFaceColor','b')
plot(conversionsSTR(icSTR).xB_extra,conversionsSTR(icSTR).yB_extra,'or','markerFaceColor','r')
title(sprintf('min dist: %0.2f, dt : %0.2f, dAB : %0.2f',...
    conversionsSTR(icSTR).mindist,...
    conversionsSTR(icSTR).tminB-conversionsSTR(icSTR).tmaxA,...
    conversionsSTR(icSTR).dAB))
%%
figure('defaultAxesFontSize',20,'position',[1062 452 837  529]), box on
h01 = histogram([trajArray_CAM1_BACKUP.dsSUMwindow],'Normalization','probability',...
    'faceAlpha',0.3,'facecolor','r');
hold on
pause(1)
h02 = histogram([trajArray_CAM1.dsSUMwindow],'Normalization','probability',...
    'faceAlpha',0.3,'facecolor','b');
legend('before','current')
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


%%
%% DARCY02_findTracks

function [trajArray_CAM1,tracks_CAM1,CCout,M,filename] = DARCY02_findTracks(allExpeStrct,iexpe,ifile,maxdist,longmin,varargin)

% 1. load image
% 2. subtract mean of the image sequence
% 3. find particles positions on all images

%fprintf('line 1308 \n')

dofigures = 'no';
if numel(varargin)
    dofigures = 'no';
    if strcmp(varargin{2},'yes')
        dofigures = varargin{2};
    end
end

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
        axis([0 size(ImMean,1) 0 size(ImMean,2)])
        h = patch('Faces',[1:4],'Vertices',[0 0;size(ImMean,1) 0;size(ImMean,1) size(ImMean,2);0 size(ImMean,2)]);
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
        htrck = plot(Xtck',Ytck','-k','lineWidth',1);
end
%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%


end
