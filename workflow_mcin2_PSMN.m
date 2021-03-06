%% Workflow_mcin2PSMN


%% STEP 0 - Defining paths using list of experiments stored in sturture allExpeStrct
fprintf('define folders and experiment name\n')


iexpe = 6;

allExpeStrct(iexpe).type        = 'experiment'; % experiment / calibration
allExpeStrct(iexpe).name        = 'expe_2021_06_17_run02';
allExpeStrct(iexpe).inputFolder = ...
    strcat('E:\manipIFPEN\expe_2021_06_17_onePlane\run02\');
allExpeStrct(iexpe).analysisFolder = ...
    strcat('D:\IFPEN\analysisExperiments\analysis_expe_2021_06_17\run02\');
allExpeStrct(iexpe).CalibFile = ...
    strcat('E:\manipIFPEN\expe_2021_06_09_calibration\calibrationImages\calib.mat');
allExpeStrct(iexpe).centerFinding_th = 5; % automatiser la définition de ces paramètres?
allExpeStrct(iexpe).centerFinding_sz = 2; % automatiser la définition de ces paramètres?
allExpeStrct(iexpe).maxdist = 3;          % for Benjamin tracks function:
% max distances between particules from frame to frame
allExpeStrct(iexpe).longmin = 8;         % for Benjamin tracks function:
% minimum number of points of a trajectory




%% STEP 1 - findTracks
iexpe = 6; % 1 / 2 / 3

allresults = struct();
cd(allExpeStrct(iexpe).analysisFolder)
file_log_ID = fopen(sprintf('log_%s.txt',allExpeStrct(iexpe).name), 'a');
 
for iplane = 1 : 200

    close all

iSeqa = iplane*2-1;
iSeqb = iplane*2;

cPlane_i = clock;
c1i = clock; fprintf('starts looking for trajectories at %0.2dh%0.2dm\n',c1i(4),c1i(5))

allTraj = struct();

maxdist = allExpeStrct(iexpe).maxdist;
longmin = allExpeStrct(iexpe).longmin;
for iSeq = iSeqa:iSeqb %35:36  % loop on images sequences
    clear trajArray_loc tracks_loc CCout
    [trajArray_loc,tracks_loc,CCout,M,filenamePlane] = ...,
        DARCY02_findTracks(allExpeStrct,iexpe,iSeq,maxdist,longmin,'figures','yes');
    allTraj(iSeq).trajArray = trajArray_loc;
    allTraj(iSeq).tracks    = tracks_loc;
    allTraj(iSeq).CC        = CCout;
    
end
fprintf('done \n')

c1f = clock; fprintf('done looking for trajectories at %0.2dh%0.2dm in %0.0f s \n',c1f(4),c1f(5), etime(c1f,c1i))

% STEP 2 - start robust estimation

c2i = clock; fprintf('starts fitgeotrans between the two cameras at %0.2dh%0.2dm\n',c2i(4),c2i(5))

%close all
CalibFile = allExpeStrct(iexpe).CalibFile;

him = size(M,1); % 1152;
wim = size(M,2); % 1152;
clear CCtemp CC1 CC2 totalnFrames
CC1 = allTraj(iSeqa).CC; % CCtemp.CC;
CC2 = allTraj(iSeqb).CC; % CCtemp.CC;
totalnFrames = size(CC1,2);

% STEP 3 - removing the NaNs for all t
for it = 1 : size(CC1,2)
    ikill = [];
    for ip = 1 : size(CC1(it).X,1)
        if isnan(CC1(it).X(ip)) || isnan(CC1(it).Y(ip))
            ikill = [ikill,ip];
        end
    end
    CC1(it).X(ikill) = [];
    CC1(it).Y(ikill) = [];
    clear ikill
    ikill = [];
    for ip = 1 : size(CC2(it).X,1)
        if isnan(CC2(it).X(ip)) || isnan(CC2(it).Y(ip))
            ikill = [ikill,ip];
        end
    end
    CC2(it).X(ikill) = [];
    CC2(it).Y(ikill) = [];
end
%  normxcorr2 - we build the images

ACC1 = zeros(him,wim,'uint8');
ACC2 = zeros(him,wim,'uint8');
for it = 1 : totalnFrames
    for ip = 1 : length(CC1(it).X)
        xim1 = round(CC1(it).X(ip));
        yim1 = round(CC1(it).Y(ip));
        ACC1(yim1,xim1) = ACC1(yim1,xim1) + 255;
    end
    for ip = 1 : length(CC2(it).X)
        xim2 = round(CC2(it).X(ip));
        yim2 = round(CC2(it).Y(ip));
        ACC2(yim2,xim2) = ACC1(yim2,xim2) + 255;
    end
end

hcam01 = figure;
imagesc(20*ACC1)%, colormap gray

hcam02 = figure;
imagesc(20*ACC2)%, colormap gray


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% normxcorr2 pass 01 (on a large window)
% close all
monPos = get(0,'MonitorPositions');
wmon=monPos(end,3); hmon=monPos(end,4);
hcam01 = figure;
imagesc(20*ACC1)%, colormap gray
title('Camera1')
set(gcf,'position',[150  hmon/2-100 wmon/3 hmon/2]);
hcam02 = figure;
imagesc(20*ACC2)%, colormap gray
title('Camera2')
set(gcf,'position',[200+wmon/3 hmon/2-100 wmon/3 hmon/2]);


% xm,ym : fixed points in camera 1
c = clock; fprintf('start at %0.2dh%0.2dm\n',c(4),c(5))
filterOrder = 10;

% first pass
xm = 00+round(wim/2);
ym = 00+round(him/2);
wsub = 250; %round(0.25*mean(xm,ym)); % width correlation template image
[xoffSet,yoffSet] = imageCorrelation(xm,ym,ACC1,ACC2,wsub,filterOrder);

figure(hcam01), hold on
drawrectangle(gca,'Position',[xm-wsub,ym-wsub,2*wsub,2*wsub], ...
    'FaceAlpha',0,'Color','b');
figure(hcam02), hold on
drawrectangle(gca,'Position',[xoffSet-wsub,yoffSet-wsub,2*wsub,2*wsub], ...
    'FaceAlpha',0,'Color','r');
dxPass01 =   xoffSet-xm;
dyPass01 =   yoffSet-ym;
R = (dxPass01^2+dyPass01^2)^(1/2);
c = clock; fprintf('finished at %0.2dh%0.2dm\n',c(4),c(5))

sub01 = imcrop(ACC1,[xm-wsub,ym-wsub,2*wsub,2*wsub]);
hcam01sub = figure;
imagesc(sub01)%, colormap gray
title('Camera1 crop')
set(gcf,'position',[265    90   431   360]);

sub02 = imcrop(ACC2,[xoffSet-wsub,yoffSet-wsub,2*wsub,2*wsub]);
hcam02sub = figure;
imagesc(sub02)%, colormap gray
title('Camera2 crop')
set(gcf,'position',[765    90   431   360]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ROBUST ESTIMATION PART 1.3 normxcorr2 pass 02 (on a small window)

wti = 300; % width template images
wstep = 100; % step for sampling the image
nPartMin = 100; % minimum number of particles to calculate the correlation
tmpl_IM_tStr = struct(); % structure storing information on template images

% cut the image in a lot of small images
hcam01 = figure('defaultAxesFontSize',20);
imagesc(20*ACC1)%, colormap gray
title('Camera1'), hold on
clear nCol nRow
nCol = wim / wstep;
nLin = him / wstep;
iti = 0;
for iCol = 1 : nCol
    for iLin = 1 : nLin
        clear xc yc
        xc = round(1 + round(wstep/2) + (iCol-1)*wstep*nCol/floor(nCol));
        yc = round(1 + round(wstep/2) + (iLin-1)*wstep*nLin/floor(nLin));
        if xc-wti/2 < 0 || yc-wti/2 < 0 || xc+wti/2 > wim || yc+wti/2 > him
            continue
        end
        iti = iti + 1;
        tmpl_IM_tStr(iti).x = xc;
        tmpl_IM_tStr(iti).y = yc;
        clear subIm
        subIm =  ACC1(yc-wti/2:yc+wti/2,xc-wti/2:xc+wti/2);
        tmpl_IM_tStr(iti).subIm = subIm;
        tmpl_IM_tStr(iti).meanSubIm = mean(subIm(:));
        if tmpl_IM_tStr(iti).meanSubIm*(101*101)/255 > nPartMin  && ...
                (1.5*dxPass01) + xc + wti/2 > 0 && ...
                (1.5*dyPass01) + yc + wti/2 > 0
            tmpl_IM_tStr(iti).correlable = 1;
            pcol = 'g';
        else
            tmpl_IM_tStr(iti).correlable = 0;
            pcol = 'b';
        end
        figure(hcam01)
        clear xp yp
        xp = .5*[-1  1  1 -1 -1]*wti+tmpl_IM_tStr(iti).x;
        yp = .5*[-1 -1  1  1 -1]*wti+tmpl_IM_tStr(iti).y;
        patch('xdata',xp,'ydata',yp,'faceColor','none','faceAlpha',.3,'edgeColor',pcol)
        pause(.2)
        
        if tmpl_IM_tStr(iti).correlable == 1
            clear xm ym xoffSet yoffSet
            xm = tmpl_IM_tStr(iti).x;
            ym = tmpl_IM_tStr(iti).y;
            [xoffSet,yoffSet] = imageCorrelation(xm,ym,ACC1,ACC2,...
                round(wti/2),filterOrder,'cleanC',dxPass01,dyPass01,150);
            tmpl_IM_tStr(iti).xoffSet = xoffSet;
            tmpl_IM_tStr(iti).yoffSet = yoffSet;
            figure(hcam01)
            if abs(xoffSet-xm- dxPass01)<100 && abs(yoffSet-ym- dyPass01)<100
                quiver(xm,ym,xoffSet-xm,yoffSet-ym,'-r','lineWidth',2)
            else
                tmpl_IM_tStr(iti).correlable = 0;
                quiver(xm,ym,xoffSet-xm,yoffSet-ym,'--r','lineWidth',1)
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% build tform1
clear fixedPoints movingPoints
corOK = ([tmpl_IM_tStr.correlable] == 1);
fixedPoints  = [[tmpl_IM_tStr(corOK).x]',      [tmpl_IM_tStr(corOK).y]'];
movingPoints = [[tmpl_IM_tStr(corOK).xoffSet]',[tmpl_IM_tStr(corOK).yoffSet]'];

% figure
% clear X Y U V
% X = fixedPoints(:,1);
% Y = fixedPoints(:,2);
% U = movingPoints(:,1)-X;
% V = movingPoints(:,2)-Y;
% quiver(X,Y,U,V)
% set(gca,'ydir','reverse')

transformationType = 'affine';
tform1 = fitgeotrans(movingPoints,fixedPoints,transformationType);

% check some points
[X,Y] = transformPointsForward(tform1,1010,583);  % check some points
[X,Y] = transformPointsForward(tform1,0,0);       % check the change of (x0,y0)

ACC2tformed = imwarp(ACC2,tform1, 'OutputView', imref2d( size(ACC1) ));

falseColorOverlay = imfuse( 40*ACC1, 40*ACC2tformed);
imshow( falseColorOverlay, 'initialMagnification', 'fit');
set(gcf,'position',[ 189         122        1058         858])
% figure to check the tform1
% figure, hold on, box on
% for it = 1 : size(CC1,2)
%     %pause(.1)
%     inputPoints = [CC2(it).X;CC2(it).Y]';
%     PointsC1 = [CC1(it).X;CC1(it).Y]';
%     [X,Y] = transformPointsForward(tform,inputPoints(:,1),inputPoints(:,2));
%
%     plot(X,Y,'or')
%     plot(PointsC1(:,1),PointsC1(:,2),'ob')
%     pause(.1)
%     axis([110 210 700 900])
% end

c2f = clock; fprintf('done fitgeotrans between the two cameras %0.2dh%0.2dm in %0.0f s \n',c2f(4),c2f(5), etime(c2f,c2i))

% STEP 4 - track particles in 2D on each camera
c4i = clock; fprintf('starts tracking particles at %0.2dh%0.2dm\n',c4i(4),c4i(5))

% from BLP TRAJECTOIRE 2D
clear part_cam1 part_cam2 part_cam2RAW
for it = 1 : size(CC1,2)
    part_cam1(it).pos(:,1) = [CC1(it).X]; % out_CAM1(:,1);
    part_cam1(it).pos(:,2) = [CC1(it).Y]; % out_CAM1(:,2);
    part_cam1(it).pos(:,3) = ones(length([CC1(it).X]),1)*it;
    part_cam1(it).intensity = 0; %mI;
    
    clear cam2X cam2Y
    part_cam2RAW(it).pos(:,1) = [CC2(it).X]; % out_CAM1(:,1);
    part_cam2RAW(it).pos(:,2) = [CC2(it).Y]; % out_CAM1(:,2);
    part_cam2RAW(it).pos(:,3) = ones(length([CC2(it).X]),1)*it;
    part_cam2RAW(it).intensity = 0; %mI;
    
    [cam2X,cam2Y] = transformPointsForward(tform1,CC2(it).X,CC2(it).Y);
    part_cam2(it).pos(:,1) = [cam2X]; % out_CAM1(:,1);
    part_cam2(it).pos(:,2) = [cam2Y]; % out_CAM1(:,2);
    part_cam2(it).pos(:,3) = ones(length([cam2X]),1)*it;
    part_cam2(it).intensity = 0; %mI;
end

maxdist = 3;
longmin = 5;
[trajArray_CAM1,tracks_CAM1]          = TAN_track2d(part_cam1,maxdist,longmin);
[trajArray_CAM2RAW,tracks_CAM2RAW]    = TAN_track2d(part_cam2RAW,maxdist,longmin);

[trajArray_CAM2,tracks_CAM2]=TAN_track2d(part_cam2,maxdist,longmin);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% keep only long trajectories
ltraj = 5; % = 10
clear ikill01 ikill02
for itraj1 = 1 : length(trajArray_CAM1)
    lt1(itraj1) = size(trajArray_CAM1(itraj1).track,1) ;
end
ikill01 = find(lt1<ltraj);
% figure
% histogram(lt1,[0:2:1000])
% hold on
% plot(10*[1 1],[0 400],'--k')

for itraj2 = 1 : length(trajArray_CAM2)
    lt2(itraj2) = size(trajArray_CAM2(itraj2).track,1) ;
end
ikill02 = find(lt2<ltraj);
% figure
% histogram(lt2)

trajArray_CAM1(ikill01)=[];
trajArray_CAM2(ikill02)=[];

c4f = clock; fprintf('done tracking particles at %0.2dh%0.2dm in %0.0f s \n',c4f(4),c4f(5), etime(c4f,c4i))

%%%%% %%%%% %%%%% %%%%% %%%%%
%%%%% %%%%% %%%%% %%%%% %%%%%
% STEP 5 - associate trajectories
c5i = clock; fprintf('starts associating trajectories at %0.2dh%0.2dm\n',c5i(4),c5i(5))

listMatchedTracks = struct(); % list of potential tracks
ilist = 0;
for itrajCam0 = 1 : length(trajArray_CAM1)
    if (~mod(itrajCam0,100) == 1) || (itrajCam0==1)
        fprintf('index trajectory: %0.0f / %0.0f \n',itrajCam0,length(trajArray_CAM1))
    end
    %fprintf('itrajCam0 : %0.0f \n',itrajCam0)
    [itrajCam1,dtraj,listPotTracks,prelist] = DARCY02_matchingTracks(itrajCam0,trajArray_CAM1,trajArray_CAM2RAW,tform1);
    if itrajCam1
        ilist = ilist +1;
        listMatchedTracks(ilist).trajcam0 = itrajCam0;
        listMatchedTracks(ilist).trajcam1 = itrajCam1;
    end
end

c5f = clock; fprintf('trajectories associated at %0.2dh%0.2dm in %0.0f s \n',c5f(4),c5f(5), etime(c5f,c5i))

%%%%% %%%%% %%%%% %%%%% %%%%%
%%%%% %%%%% %%%%% %%%%% %%%%%
% STEP 6 - Crossing the rays
% c = clock; fprintf('on croise les doigts at %0.2dh%0.2dm\n',c(4),c(5))
% 
% Ttype = 'T3';
% tic
% h3D = figure('defaultAxesFontSize',20); box on, hold on, view(3)
% xlabel('x'), ylabel('y'), zlabel('z')
% 
% calibTemp = load(CalibFile,'calib'); calib = calibTemp.calib;
% CalibFileCam1 = calib(:,1);
% CalibFileCam2 = calib(:,2);
% 
% for iP = 1 : 1 : length(structPotentialPairs)
%     if structPotentialPairs(iP).d < 15
%         fprintf('progress: %0.0f / %0.0f \n',iP,length(structPotentialPairs) )
%         if 1%structPotentialPairs(iP).matched == 1
%             itrj01 = structPotentialPairs(iP).trajCAM01;
%             itrj02 = structPotentialPairs(iP).trajCAM02;
%             
%             clear x01 y01 x02 y02 x02incam01 y02incam01
%             x01 = trajArray_CAM1(itrj01).track(structPotentialPairs(iP).tCAM01,1);
%             y01 = trajArray_CAM1(itrj01).track(structPotentialPairs(iP).tCAM01,2);
%             x02incam01 = trajArray_CAM2(itrj02).track(structPotentialPairs(iP).tCAM02,1);
%             y02incam01 = trajArray_CAM2(itrj02).track(structPotentialPairs(iP).tCAM02,2);
%             [ x02, y02] = transformPointsInverse(tform1,x02incam01,y02incam01);
%             
%             for ixy = 1 : length(x01)
%                 
%                 x_pxC1 = x01(ixy);
%                 y_pxC1 = y01(ixy);
%                 x_pxC2 = x02(ixy);
%                 y_pxC2 = y02(ixy);
%                 
%                 crossP = crossRaysonFire(CalibFileCam2,CalibFileCam1,x_pxC1,y_pxC1,x_pxC2,y_pxC2,Ttype);
%                 
%                 if length(crossP)>0
%                     %     figure(h3D), hold on
%                     %     plot3(crossP(1),crossP(2),crossP(3),'og')
%                     structPotentialPairs(iP).x3D(ixy) = crossP(1);
%                     structPotentialPairs(iP).y3D(ixy) = crossP(2);
%                     structPotentialPairs(iP).z3D(ixy) = crossP(3);
%                 end
%             end
%         end
%     end
% end
% 
% %axis equal
% c = clock; fprintf('rays crossed at %0.2dh%0.2dm\n',c(4),c(5))

%%%%% %%%%% %%%%% %%%%% %%%%%
%%%%% %%%%% %%%%% %%%%% %%%%%
% cross rays with trajectories found with DARCY02_matchingTracks
ci = clock; fprintf('start crossing rays at %0.2dh%0.2dm\n',c(4),c(5))

someTrajectories = struct();
Ttype = 'T3';
calibTemp = load(CalibFile,'calib'); calib = calibTemp.calib;
CalibFileCam1 = calib(:,1);
CalibFileCam2 = calib(:,2);

for iselTraj = 1 : size(listMatchedTracks,2)

    itraj1 = listMatchedTracks(iselTraj).trajcam0;
    itraj2 = listMatchedTracks(iselTraj).trajcam1;
%     figure
%     plot([trajArray_CAM1(itraj1).track(:,1)],[trajArray_CAM1(itraj1).track(:,2)],'-ob')
%     figure
%     plot([trajArray_CAM2RAW(itraj2).track(:,1)],[trajArray_CAM2RAW(itraj2).track(:,2)],'-ob')
%     
    someTrajectories(iselTraj).itraj1 = itraj1;
    someTrajectories(iselTraj).itraj2 = itraj2;
    
    % cross the two choosen rays
    clear x01 y01 x02 y02 x02incam01 y02incam01
    tminCAM01 = min(trajArray_CAM1(itraj1).track(:,3));
    tmaxCAM01 = max(trajArray_CAM1(itraj1).track(:,3));
    tminCAM02 = min(trajArray_CAM2RAW(itraj2).track(:,3));
    tmaxCAM02 = max(trajArray_CAM2RAW(itraj2).track(:,3));
    [A,B,C] = intersect([tminCAM01:tmaxCAM01],[tminCAM02:tmaxCAM02]);
    
    A;
    if A
        clear x01 y01 x02 y02
        x01 = trajArray_CAM1(itraj1).track(min(B):max(B),1);
        y01 = trajArray_CAM1(itraj1).track(min(B):max(B),2);
        x02 = trajArray_CAM2RAW(itraj2).track(min(C):max(C),1);
        y02 = trajArray_CAM2RAW(itraj2).track(min(C):max(C),2);
        
        clear x_pxC1 y_pxC1 x_pxC2 y_pxC2
        for ixy = 1 : length(x01)
            
            x_pxC1 = x01(ixy);
            y_pxC1 = y01(ixy);
            x_pxC2 = x02(ixy);
            y_pxC2 = y02(ixy);
            
            [crossP,D] = crossRaysonFire(CalibFileCam1,CalibFileCam2,x_pxC1,y_pxC1,x_pxC2,y_pxC2,Ttype);
            if length(crossP)>0
                someTrajectories(iselTraj).x3D(ixy) = crossP(1);
                someTrajectories(iselTraj).y3D(ixy) = crossP(2);
                someTrajectories(iselTraj).z3D(ixy) = crossP(3);
                someTrajectories(iselTraj).D(ixy) = D;
            end
        end
    end
    
    
end

ce = clock; fprintf('done crossing rays at %0.2dh%0.2dm in %0.0f s \n',c(4),c(5), etime(ce,ci))

%%%%%%%%%
hist3D = [];
histD  = [];
% plot the result
figure, hold on
for itraj3D = 1 : size(someTrajectories,2)
    plot3([someTrajectories(itraj3D).x3D],...
        [someTrajectories(itraj3D).y3D],...
        [someTrajectories(itraj3D).z3D],'og')
     hist3D = [hist3D,[someTrajectories(itraj3D).z3D]]; 
     histD  = [histD, [someTrajectories(itraj3D).D]]; 
end
view(3)
xlabel('x')
ylabel('y')
zlabel('z')
axis equal
title(sprintf('plane: %0.2d, file : %s',iplane,filenamePlane)) 

figure
histogram(hist3D)
title(sprintf('plane: %0.2d, file : %s',iplane,filenamePlane)) 

figure
histogram(histD)
title(sprintf('plane: %0.2d, file : %s',iplane,filenamePlane)) 

allresults(iplane).someTrajectories = someTrajectories;
allresults(iplane).hist3D = hist3D;
allresults(iplane).histD = histD;
allresults(iplane).trajArray_CAM1 = trajArray_CAM1;
allresults(iplane).trajArray_CAM2RAW = trajArray_CAM2RAW;
allresults(iplane).listMatchedTracks = listMatchedTracks;
allresults(iplane).tform1 = tform1;
allresults(iplane).filenamePlane = filenamePlane;

cd(allExpeStrct(iexpe).analysisFolder)
save('allResults_auFilDeLEau.mat','allresults')

cPlane_f = clock;
fprintf(file_log_ID, 'just run plane %0.3d in %0.0f s \n', iplane , etime(cPlane_f,cPlane_i) );

end

fclose(file_log_ID)
%% figure for a whole experiment
figure('defaultAxesFontSize',20), box on, hold on
edges = [0:0.1:30];
for iplane = 1 : 100
    clear hist3D N
    hist3D = allresults(iplane).hist3D;
    [N] = histcounts(hist3D,edges);
    plot(edges(1:end-1),N,'-','lineWidth',4)
end

% plot the result
figure('defaultAxesFontSize',20), box on, hold on
for iplane = 1 : 100
    someTrajectories = allresults(iplane).someTrajectories;
    for itraj3D = 1 : size(someTrajectories,2)
        Test = mean([allresults(iplane).someTrajectories(itraj3D).D  ],'omitnan') > 1.5 || ...
            mean([allresults(iplane).someTrajectories(itraj3D).z3D  ],'omitnan') < 18 ;
        if ~Test
        plot3([someTrajectories(itraj3D).x3D],...
            [someTrajectories(itraj3D).y3D],...
            [someTrajectories(itraj3D).z3D],'.b')
        end
    end
end
view(3)
xlabel('x')
ylabel('y')
zlabel('z')
axis equal

% xlim([10 30])
% ylim([0 20])
zlim([15 30])

% xlim([10 30])
% ylim([0 20])
% zlim([15 30])

%% comparing mean z of a trajectory with crossing distance between rays D
figure('defaultAxesFontSize',20), grid on, box on, hold on
for itrj = 1 : 100
clear meanD meanZ
for it = 1 : size(allresults(itrj).someTrajectories  ,2)
    meanZ(it) = mean([allresults(itrj).someTrajectories(it).z3D  ]);
    meanD(it) = mean([allresults(itrj).someTrajectories(it).D  ]);
end
plot(meanZ,meanD,'ok')
end
xlabel('meanZ')
ylabel('meanD')

mean(meanZ,'omitnan')
var(meanZ,'omitnan')
%% IV. Faire de belles images
% voxels of 1mm^3, check if they contain particules or not

clear allTracersXYZ allTracersX allTracersY allTracersZ
allTracersX = [];
allTracersY = [];
allTracersZ = [];
for iplane = 7 : 21
    clear someTrajectories
    someTrajectories = allresults(iplane).someTrajectories;
    for itraj3D = 1 : size(someTrajectories,2)
        allTracersX = [allTracersX , someTrajectories(itraj3D).x3D];
        allTracersY = [allTracersY , someTrajectories(itraj3D).y3D];
        allTracersZ = [allTracersZ , someTrajectories(itraj3D).z3D];
    end
end
allTracersXYZ = [allTracersX;allTracersY;allTracersZ];


%%
voxBeads = struct();
ivb = 0; % index voxel beads
wVox = 1; % edge length of voxels (mm)
listX = -15:15;
listY = -20:wVox:10;
listZ = 0:wVox:20;
for ix = 1:length(listX)-1
    for iy = 1:length(listY)-1
        for iz =  1:length(listZ)-1
            ivb = ivb + 1;
            % compute number of particles in the voxel
            clear x y z
            xA = listX(ix); xB = listX(ix+1);
            x = [xA,xA,xB,xB,xA,xA,xB,xB,(xA+xB)/2]';
            yA = listY(iy); yB = listY(iy+1);
            y = [yA,yB,yA,yB,yA,yB,yA,yB,(yA+yB)/2]';
            zA = listZ(iz); zB = listZ(iz+1);
            z = [zA,zA,zA,zA,zB,zB,zB,zB,(zA+zB)/2]';
            voxBeads(ivb).x = x;
            voxBeads(ivb).y = y;
            voxBeads(ivb).z = z;
            tri = delaunayn([x y z]); % Generate delaunay triangulization
            tn = tsearchn([x y z], tri, allTracersXYZ'); % Determine which triangle point is within
            IsInside = ~isnan(tn);
            fprintf('ivb : %0.0f, n° inside: %0.0f \n',ivb,sum(IsInside))
            
            voxBeads(ivb).nParticles = sum(IsInside);
            if sum(IsInside) < 10
                
                voxBeads(ivb).Bead = 1;
            else
                
                voxBeads(ivb).Bead = 0;
            end
        end
    end
end
%%
%%
colrP = jet(size(allresults,2));
figure('defaultAxesFontSize',20), box on, hold on
view(-36,21)
xlabel('x')
ylabel('y')
zlabel('z')
axis equal
zlim([-20 60 ])
for iplane = 1 : size(allresults,2)
    someTrajectories = allresults(iplane).someTrajectories;
    for itraj3D = 1 : size(someTrajectories,2)
        plot3([someTrajectories(itraj3D).x3D],...
            [someTrajectories(itraj3D).y3D],...
            [someTrajectories(itraj3D).z3D],'o','markerFaceColor',colrP(iplane,:),'MarkerEdgeColor',colrP(iplane,:))
    end
    pause(2)
end


pause(1)
%figure('defaultAxesFontSize',20), box on, hold on
%view(3)
%axis equal

for ivb = 1 : length(voxBeads)
    x = voxBeads(ivb).x;
    y = voxBeads(ivb).y;
    z = voxBeads(ivb).z;
    if    voxBeads(ivb).Bead == 0
        [k1] = convhull(x,y,z,'Simplify',true);
        trisurf(k1,x,y,z,...
            'FaceColor',[min(10*voxBeads(ivb).nParticles,255),20,5]/255,...
            'edgeColor','none')
        pause(.1)
    end
end

%% speed in voxels

% build data including all planes
voxData = struct();
histx3D = [];
histy3D = [];
histz3D = [];
allTrajAllPlanes = struct(); iatap = 0;
for iplane = 1 : size(allresults,2)
    someTrajectories = allresults(iplane).someTrajectories;
    for itraj3D = 1 : size(someTrajectories,2)
        Test = mean([allresults(iplane).someTrajectories(itraj3D).D  ],'omitnan') > 1.5 || ...
            mean([allresults(iplane).someTrajectories(itraj3D).z3D  ],'omitnan') < 18 ;
        if ~Test
        iatap =  iatap + 1;
        allTrajAllPlanes(iatap).itraj1 = someTrajectories(itraj3D).itraj1;
        allTrajAllPlanes(iatap).itraj2 = someTrajectories(itraj3D).itraj2;
        allTrajAllPlanes(iatap).x3D = someTrajectories(itraj3D).x3D;
        allTrajAllPlanes(iatap).y3D = someTrajectories(itraj3D).y3D;
        allTrajAllPlanes(iatap).z3D = someTrajectories(itraj3D).z3D;
        allTrajAllPlanes(iatap).D = someTrajectories(itraj3D).D;
        allTrajAllPlanes(iatap).iplane = iplane;
        histx3D = [histx3D,[allTrajAllPlanes(iatap).x3D]];
        histy3D = [histy3D,[allTrajAllPlanes(iatap).y3D]];
        histz3D = [histz3D,[allTrajAllPlanes(iatap).z3D]];
        end
    end
end
figure, histogram(histx3D), title('distribution along x')
figure, histogram(histy3D), title('distribution along y')
figure, histogram(histz3D), title('distribution along z')

%%

wVox = 2
listX = -25 : wVox : 35;
listY = -20 : wVox : 25;
listZ =   19 : wVox : 27;
% 
% listX = -50 : wVox : 50;
% listY = -50 : wVox : 50;
% listZ = -20 : wVox : 65;

for ix = 1:length(listX)-1
    for iy = 1:length(listY)-1
        for iz =  1:length(listZ)-1
            voxData(ix,iy,iz).x = (listX(ix) + listX(ix+1))/2;
            voxData(ix,iy,iz).y = (listY(iy) + listY(iy+1))/2;
            voxData(ix,iy,iz).z = (listZ(iz) + listZ(iz+1))/2;
            voxData(ix,iy,iz).v = [];
            voxData(ix,iy,iz).vx = [];
            voxData(ix,iy,iz).vy = [];
            voxData(ix,iy,iz).vz = [];
        end
    end
end
% à détailler pour rigueur géométrique
vStep = 2 % 10

tic
for itraj3D = 1 : 1 : size(allTrajAllPlanes,2)
    clear x3D y3D z3D
    x3D = [allTrajAllPlanes(itraj3D).x3D];
    y3D = [allTrajAllPlanes(itraj3D).y3D];
    z3D = [allTrajAllPlanes(itraj3D).z3D];
    for is = 1 : vStep : length(x3D)-vStep
        % find if it goes to a voxel: 
        % is x3D, y3D and z3D in listX listY and listZ and where?
        [mix,ix] = mink(abs(listX-(x3D(is)+x3D(is+vStep))/2),2);
        [miy,iy] = mink(abs(listY-(y3D(is)+y3D(is+vStep))/2),2);
        [miz,iz] = mink(abs(listZ-(z3D(is)+z3D(is+vStep))/2),2);
        if mix(1)<1 && miy(1)<1 && miz(1)<1
           % calculate speed and store it in voxData
           clear v vx vy vz
           vx = x3D(is+vStep)-x3D(is);
           vy = y3D(is+vStep)-y3D(is);
           vz = z3D(is+vStep)-z3D(is);
           v = norm( [ vx vy vz ] );
           IX = max(1,min(ix(1),length(listX)-1));
           IY = max(1,min(iy(1),length(listY)-1));
           IZ = max(1,min(iz(1),length(listZ)-1));
           voxData(IX,IY,IZ).v  = [voxData(IX,IY,IZ).v ,v];   
           voxData(IX,IY,IZ).vx = [voxData(IX,IY,IZ).vx,vx];   
           voxData(IX,IY,IZ).vy = [voxData(IX,IY,IZ).vy,vy];   
           voxData(IX,IY,IZ).vz = [voxData(IX,IY,IZ).vz,vz];   
        end
    end
end

for ix = 1:length(listX)-1
    for iy = 1:length(listY)-1
        for iz =  1:length(listZ)-1
            
            voxData(ix,iy,iz).U = mean([voxData(ix,iy,iz).vx]);
            voxData(ix,iy,iz).V = mean([voxData(ix,iy,iz).vy]);
            voxData(ix,iy,iz).W = mean([voxData(ix,iy,iz).vz]);
        end
    end
end



clear normU U V W X Y Z
U = zeros(size(voxData),'double');
V = zeros(size(voxData),'double');
W = zeros(size(voxData),'double');
for ix = 1:length(listX)-1
    for iy = 1:length(listY)-1
        for iz = 1:length(listZ)-1
            X(ix,iy,iz) = voxData(ix,iy,iz).x;
            Y(ix,iy,iz) = voxData(ix,iy,iz).y;
            Z(ix,iy,iz) = voxData(ix,iy,iz).z;
            if ~isnan(voxData(ix,iy,iz).U)
            U(ix,iy,iz) = max(-.4,min(.4,voxData(ix,iy,iz).U));
            V(ix,iy,iz) = max(-.4,min(.4,voxData(ix,iy,iz).V));
            W(ix,iy,iz) = max(-1,min(1,voxData(ix,iy,iz).W));
%             U(ix,iy,iz) = voxData(ix,iy,iz).U;
%             V(ix,iy,iz) = voxData(ix,iy,iz).V;
%             W(ix,iy,iz) = voxData(ix,iy,iz).W;
            normU(ix,iy,iz) = norm([U(ix,iy,iz),V(ix,iy,iz),W(ix,iy,iz)]);
            end
        end
    end
end

%% 3D quiver 
close all
h3Dquiver = figure('defaultAxesFontSize',20,'position',[53   410   914   555]); box on, hold on
quiver3(X,Y,Z,U,V,W)
xlabel('x')
ylabel('y')
zlabel('z')
view(3)
% ylim([ -19 -18])
% zlim([ 5 20])
axis equal

%% from coordinate xP yP zP and normal uP vP wP define a plane
% then show values of vBox in this plane 

xP =  0;
yP = -5;
zP = 10;
uP = 0;
vP = 0;
wP = 1;
% define the two vectors perpendicular to uPvPwP:
e10P = 1;
e01P = 1;
figure(h3Dquiver)
zlim([-15 0])
view(3)
xlim([-20 20])
ylim([-15 30])
% plot3(xP,yP,zP,'or')
%h2Dplane = figure('defaultAxesFontSize',20,'position',[1000 100 700 700]);

%%
[Xs,Ys,Zs] = meshgrid(-2:.2:2);
V = X.*exp(-X.^2-Y.^2-Z.^2);
%%
x = min(X(:)) : 1 : max(X(:));
y = min(Y(:)) : 1 : max(Y(:));
z = min(Z(:)) : 1 : max(Z(:));
[X,Y,Z] = meshgrid(x,y,z);
% close all
xslice = [1];   
yslice = [-10];
zslice = 10;
slice(X,Y,Z,normU,xslice,yslice,zslice)
xlabel('x')
ylabel('y')
zlabel('z')
%%
point = [1,-5,10];
normal = [0,0,1];
%%
hold on
pA = [18 20 10];
pB = [24 13 10];
point = (pA + pB ) / 2;
normal = pB-pA;

[B,x,y,z] = obliqueslice(normU,point,normal);

% figure
surf(x,y,z,B,'EdgeColor','None','HandleVisibility','off');
grid on
view([-38 12])
colormap(gray)
xlabel('x-axis')
ylabel('y-axis');
zlabel('z-axis');
title('Slice in 3-D Coordinate Space')

%%
volshow(normU)
%%
cmap = parula(256);
s = sliceViewer(normU,'Colormap',cmap);
%%
cmap = parula(256);
os = orthosliceViewer(normU,'Colormap',cmap);

addlistener(os,'CrosshairMoving',@allevents);
addlistener(os,'CrosshairMoved',@allevents);
%%
%# a plane is a*x+b*y+c*z+d=0
%# [a,b,c] is the normal. Thus, we have to calculate
%# d and we're set
d = -point*normal'; %'# dot product for less typing

%# create x,y
[xx,yy]=ndgrid(1:10,1:10);

%# calculate corresponding z
z = (-normal(1)*xx - normal(2)*yy - d)/normal(3);

%# plot the surface
figure
surf(xx,yy,z)

%%
nPperVoxel = zeros(1,length(voxData(:)));
for iV = 1 : length(nPperVoxel)
    nPperVoxel(iV) = length([voxData(iV).v ]);
end

figure('defaultAxesFontSize',20)
histogram( nPperVoxel)
set(gca, 'YScale', 'log')
ylim([0.9  1e4])
xlabel('quantity of measurements in 1mm3 voxels')
%%
figure
histogram(U(:))
title('x direction')

figure
histogram(V(:))
title('vertical direction (y)')

figure
histogram(W(:))
title('z direction')

%% EXPLORATION: try to show nodes and edges and hydrogel beads
delete(hpnts)
delete(hline)
delete(hbhydro)

clear xnds ynds znds verticesPatch
sNodes    = struct();
sEdge     = struct();
sHydbeads = struct();

xnds = [1.5,3,1.5,2,4,3.5,2.3,1.5,1.2];
ynds = [-.5,-2,-2,-.5,-2,-3,-3.4,-2.2,-.3];
znds = [11,11,11,13,13,13,13,13,13];
verticesPatch = [xnds',ynds',znds'];

figure(h3Dquiver)
% xlim([-1 6])
% ylim([-4 1])
% zlim([10.5 14])
% %xlim([-15 15])
% %ylim([-15 15])
% %zlim([0 30])
% xlim([-6 6])
% ylim([-10 5])
% zlim([05 20])
%hp(1) = patch('Faces',[1,2,3],'Vertices',verticesPatch,'FaceColor','red','faceAlpha',.5);

pE = [0.5,-1,15];
pF = [ -2, 3,15];
% points
hpnts(1) = plot3(2.5,-1,16,'or');
hpnts(2) = plot3(5.7, 2,18,'or');
hpnts(3) = plot3(2.5, -1.5,11.5,'or');
hpnts(4) = plot3(2, -.5,14,'or');
hpnts(5) = plot3(pE(1),pE(2),pE(3),'or');
hpnts(6) = plot3(pF(1),pF(2),pF(3),'or');
% lines
hline(1) = plot3([2.5,5.7], [-1,2],[16,18],'-r');
hline(2) = plot3([2.5,2], [-1.5,-.5],[11.5,14],'-r');
hline(2) = plot3([pE(1),pF(1)], [pE(2),pF(2)],[pE(3),pF(3)],'-r');
% beads
hbeads(1) = plot3(2,4,12,'or');
clear r x y z
r = 3;
a =  2;
b =  4;
c = 12;
[x,y,z] = sphere; 
hbhydro(1) = surf(x*r+a, y*r+b, z*r+c);
 axis equal;
 
xC = listX(16);
yC = listY(11);
zC = listZ(13);
R  =  5;
for itetha = 1 : 10 : 360
    for iPhi = -180 : 10 : 180
        x3 = xC + R*cosd(itetha) * cosd(iPhi);
        y3 = yC + R*sind(itetha) * cosd(iPhi);
        z3 = zC + R * sind(iPhi);
        plot3(x3,y3,z3,'or')
    end
end

xC = listX(26);
yC = listY(20);
zC = listZ(12);
R  =  5;
for itetha = 1 : 10 : 360
    for iPhi = -180 : 10 : 180
        x3 = xC + R*cosd(itetha) * cosd(iPhi);
        y3 = yC + R*sind(itetha) * cosd(iPhi);
        z3 = zC + R * sind(iPhi);
        plot3(x3,y3,z3,'or')
    end
end

lightangle(-45,30)
% h3Dquiver.FaceLighting = 'gouraud';
% h3Dquiver.AmbientStrength = 0.3;
% h3Dquiver.DiffuseStrength = 0.8;
% h3Dquiver.SpecularStrength = 0.9;
% h3Dquiver.SpecularExponent = 25;
% h3Dquiver.BackFaceLighting = 'unlit';
%% planes 
hp(2) = patch('Faces',[1,2,5,4],'Vertices',verticesPatch,'FaceColor','red','faceAlpha',.5);
hp(3) = patch('Faces',[2,5,6],'Vertices',verticesPatch,'FaceColor','red','faceAlpha',.5);
hp(4) = patch('Faces',[2,3,7,6],'Vertices',verticesPatch,'FaceColor','red','faceAlpha',.5);
hp(5) = patch('Faces',[3,7,8],'Vertices',verticesPatch,'FaceColor','red','faceAlpha',.5);
hp(6) = patch('Faces',[1,3,8,9],'Vertices',verticesPatch,'FaceColor','red','faceAlpha',.5);
hp(7) = patch('Faces',[1,4,9],'Vertices',verticesPatch,'FaceColor','red','faceAlpha',.5);

%% EXPLORATION lignes de courant % there is a bug 
%  -> try to do it by hand ..

x = min(X(:)) : 1 : max(X(:));
y = min(Y(:)) : 1 : max(Y(:));
z = min(Z(:)) : 1 : max(Z(:));
[Xlc,Ylc,Zlc] = meshgrid(x,y,z);

close all
h3Dquiver = figure('defaultAxesFontSize',20,'position',[53   410   914   555]); box on, hold on
quiver3(X,Y,Z,U,V,W)
xlabel('x')
ylabel('y')
zlabel('z')
view(3)
ylim([ -19 -18])
zlim([ 5 20])
axis equal

%% home made streamline
iST = 1; % index streamline
doVisual = 'yes'; % yes no

hold on
%%% %%% %%%
%%% %%% %%%
%%% %%% %%%
% initialisation
startx = 1.5;
starty = -18.5;
startz = 10.5;
startx = -0.5;
starty = 1.5;
startz = 14.5;

streamline(iST).x = startx;
streamline(iST).y = starty;
streamline(iST).z = startz;

ix = find(X(:,1,1)==startx);
iy = find(Y(1,:,1)==starty);
iz = find(Z(1,1,:)==startz);
% move the particle to the next voxel where there is data
point = [X(ix,1,1),Y(1,iy,1),Z(1,1,iz)];
normal = [  U(ix,iy,iz),...     
            V(ix,iy,iz),...
            W(ix,iy,iz)];
streamline(iST).UVW = normal;

normal =  normal / norm(normal);

%%% %%% %%%
%%% %%% %%%
%%% %%% %%%
% propagation
while(1)
    fprintf('yo % 0.2d \n',length(streamline(iST).x))
switch doVisual
    case 'yes'
        quiver3(    point(1),    point(2),    point(3),...
                10*normal(1),10*normal(2),10*normal(3),...
                'lineWidth',4)
end
% loop on the neighbouring voxels from the closest to the farthest and stop
% when UVW exists % there is 26 possible neighbours
voxNeig = struct();
ivn = 0;
for iix = 1 : 1 : 3
    for iiy = 1 : 1 : 3
        for iiz = 1 : 1 : 3
            newpoint = [X(ix + iix -2,1,1),Y(1,iy + iiy -2,1),Z(1,1,iz + iiz -2)];
            A = normal;
            B = [iix-2,iiy-2,iiz-2];
            lProj = dot(A, B, 2) ./ sum(B .* B, 2); % using https://fr.mathworks.com/matlabcentral/answers/2216-projecting-a-vector-to-another-vector
            if  (lProj<0)
                colP = 'r';
            elseif (iix == 2 && iiy == 2 && iiz == 2) %||
                
                colP = 'y';
            else
                colP = 'g';
                ivn = ivn + 1;
                voxNeig(ivn).ix = ix + iix -2;
                voxNeig(ivn).iy = iy + iiy -2;
                voxNeig(ivn).iz = iz + iiz -2;
                voxNeig(ivn).d = point_to_line(newpoint, point, point + normal);
                voxNeig(ivn).proj = lProj;

            end
            switch doVisual
                case 'yes'
                    % make a colored cube
                    ccrnsX = newpoint(1)+[-.5,+.5,-.5,+.5,-.5,+.5,-.5,+.5]; % cube corners
                    ccrnsY = newpoint(2)+[-.5,-.5,+.5,+.5,-.5,-.5,+.5,+.5]; % cube corners
                    ccrnsZ = newpoint(3)+[-.5,-.5,-.5,-.5,+.5,+.5,+.5,+.5]; % cube corners
                    ccrnsXYZ = [ccrnsX',ccrnsY',ccrnsZ'];
                   % fP = [1,2,4,3;3,4,8,7;7,8,6,5;1,2,6,5;2,4,8,6;1,3,7,5]; % faces
                   % patch('faces',fP,'vertices',ccrnsXYZ,...
                   %     'facecolor',colP,'faceAlpha',.5,'edgeColor','k')
            end
        end
    end
end
% find closest voxel and test if it contains a speed value
[~,b] = min([voxNeig.d]);
newpoint = [X(voxNeig(b).ix,1,1),Y(1,voxNeig(b).iy,1),Z(1,1,voxNeig(b).iz)];
% make a colored cube
ccrnsX = newpoint(1)+[-.55,+.55,-.55,+.55,-.55,+.55,-.55,+.55]; % cube corners
ccrnsY = newpoint(2)+[-.55,-.55,+.55,+.55,-.55,-.55,+.55,+.55]; % cube corners
ccrnsZ = newpoint(3)+[-.55,-.55,-.55,-.55,+.55,+.55,+.55,+.55]; % cube corners
ccrnsXYZ = [ccrnsX',ccrnsY',ccrnsZ'];
fP = [1,2,4,3;3,4,8,7;7,8,6,5;1,2,6,5;2,4,8,6;1,3,7,5]; % faces
patch('faces',fP,'vertices',ccrnsXYZ,...
    'facecolor','k','faceAlpha',.5,'edgeColor','k')

% test if it contains a speed value
nAttemps = size(voxNeig,2);
while nAttemps
    [~,b] = min([voxNeig.d]);
    nAttemps = nAttemps -1;
    if U(voxNeig(b).ix,voxNeig(b).iy,voxNeig(b).iz) ~= 0
        break
    else
        % fprintf('killing an element\n')
        voxNeig(b) = [];
    end
end
if nAttemps == 0
    break
end
% streamline(iST).UVW = normal;
ix = voxNeig(b).ix;
iy = voxNeig(b).iy;
iz = voxNeig(b).iz;
% move the particle to the next voxel where there is data
point = [X(ix,1,1),Y(1,iy,1),Z(1,1,iz)];
normal = [  U(ix,iy,iz),...
    V(ix,iy,iz),...
    W(ix,iy,iz)];
streamline(iST).UVW = [streamline(iST).UVW,normal];
normal =  normal / norm(normal);
streamline(iST).x = [streamline(iST).x,point(1)];
streamline(iST).y = [streamline(iST).y,point(2)];
streamline(iST).z = [streamline(iST).z,point(3)];
end
%% Je suis ICI

%%
  % EDITED, twice
C = bsxfun(@times, lenC, B)
%%
A = repmat([10,10,-10] ,[88,1]);
B = repmat([1,1,-1], [88,1]);
lenC = dot(A, B, 2) ./ sum(B .* B, 2);  % EDITED, twice
C = bsxfun(@times, lenC, B)

%% built in streamline
ci = clock; fprintf('start at %0.2dh%0.2dm\n',ci(4),ci(5))
%X 
%Y 
%Z 
%U = U;
%V = V;
%W = W;

clear startx starty startz
[startx,startz] = meshgrid( [min(Xlc(:)) : 1 : max(Xlc(:))],...
                            [min(Zlc(:)) : 1 : max(Zlc(:))]);
starty = -18.5 * ones(size(startx));

[startx,starty,startz] = meshgrid([min(X(:)) : 1 : max(X(:))],...
                                  [min(Y(:)) : 1 : max(Y(:))],...
                                  [min(Z(:)) : 1 : max(Z(:))]); 
h = streamline(Xlc,Ylc,Zlc,U,V,W,startx,starty,startz,.1);
ce = clock; fprintf('done at %0.2dh%0.2dm in %0.0f s \n',ce(4),ce(5), etime(ce,ci))
%%
xlim([5.5 9.5])
ylim([1   5  ])
zlim([14.5 18.5])
%%
tic
h = streamtube(X,Y,Z,U,V,W,startx,starty,startz);
shading interp;
camlight; 
lighting gouraud
toc

%% loop on the streamlines
% find the largest one


h3Dquiver = figure('defaultAxesFontSize',20,'position',[53   410   914   555]); box on, hold on
quiver3(X,Y,Z,U,V,W)
xlabel('x')
ylabel('y')
zlabel('z')
view(3)

lSL = []; % length of stream line
for il = 1 : length(h)
    lSL(il) = length(h(il).XData);
end
[a,b] = max(lSL);
xL = h(b).XData;
yL = h(b).YData;
zL = h(b).ZData;
plot3(xL,yL,zL,'-or','lineWidth',4)
xlim([xL(end)-4 xL(end-2)+4])
ylim([yL(end)-4 yL(end-2)+4])
zlim([zL(end)-4 zL(end-2)+4])
plot3(xL(end),yL(end),zL(end),'ob')
grid on

point = [xL(end),yL(end),zL(end)];
normal = [xL(end)-xL(end-5),yL(end)-yL(end-5),zL(end)-zL(end-5)];
normal = normal / norm(normal);

% find perpendicular vectors and plot them
quiver3(point(1),point(2),point(3),normal(1),normal(2),normal(3),'lineWidth',4)
v_perp1 = find_perp(normal)/norm(find_perp(normal));
v_perp2 = cross(normal,v_perp1)/norm(cross(normal,v_perp1));
quiver3(point(1),point(2),point(3),v_perp1(1),v_perp1(2),v_perp1(3),'lineWidth',4)
quiver3(point(1),point(2),point(3),v_perp2(1),v_perp2(2),v_perp2(3),'lineWidth',4)

% list of starting points at 1 mm from last point of the trajectory.

%%

t=(0:10:360)';
circle0=[cosd(t) sind(t) zeros(length(t),1)];
r=vrrotvec2mat(vrrotvec([0 0 1],normal));
circle=circle0*r'+repmat(point,length(circle0),1);
patch(circle(:,1),circle(:,2),circle(:,3),.5);
axis square; grid on;
%add line
line=[point;point+normr(normal)]
hold on;plot3(line(:,1),line(:,2),line(:,3),'LineWidth',5)
%% EXPLORATION : isosurface

%# create coordinates
[xx,yy,zz] = meshgrid(-15:15,-15:15,-15:15);
%# calculate distance from center of the cube
rr = sqrt(xx.^2 + yy.^2 + zz.^2);

%# create the isosurface by thresholding at a iso-value of 10
isosurface(xx,yy,zz,rr,10);

%# make sure it will look like a sphere
axis equal 
%%
close all
tic
clear rr
[xx,yy,zz] = meshgrid(  1:length(listX)-1,...
                        1:length(listY)-1,...
                        1:length(listZ)-1);
for ix = 1:length(listX)-1
    for iy = 1:length(listY)-1
        for iz = 1:length(listZ)-1
            if length([voxData(ix,iy,iz).v ]) > 1
                rr(ix,iy,iz) = 2;
            end
        end
    end
end
%# create the isosurface by thresholding at a iso-value of 10
isosurface(xx,yy,zz,rr,1);

%# make sure it will look like a sphere
axis equal 
hold on
xC = 16;
yC = 11;
zC = 13;
R  =  5;
for itetha = 1 : 2 : 360
    for iPhi = -180 : 10 : 180
        x3 = xC + R*cosd(itetha) * cosd(iPhi);
        y3 = yC + R*sind(itetha) * cosd(iPhi);
        z3 = zC + R * sind(iPhi);
        plot3(x3,y3,z3,'or')
    end
end

xC = 26;
yC = 20;
zC = 12;
R  =  5;
for itetha = 1 : 2 : 360
    for iPhi = -180 : 10 : 180
        x3 = xC + R*cosd(itetha) * cosd(iPhi);
        y3 = yC + R*sind(itetha) * cosd(iPhi);
        z3 = zC + R * sind(iPhi);
        plot3(x3,y3,z3,'or')
    end
end
xlabel('x')
ylabel('y')
zlabel('z')
box on
toc
%% EXPLORATION: best way to plot a sphere
clear r x y z
r = 25;
a =  2;
b =  4;
c = 12;
figure('defaultAxesFontSize',20)
[x,y,z] = sphere; 
surf(x*r+a, y*r+b, z*r+c);
 axis equal;
%% exploration voxel
d = [-1 1];
[x,y,z] = meshgrid(d,d,d);  % A cube
x = [x(:);0];
y = [y(:);0];
z = [z(:);0];
% [x,y,z] are corners of a cube plus the center.
X = [x(:) y(:) z(:)];
Tes = delaunayn(X)
%% exploration voxel
n = 10; % Number of vertices
theta = 2*pi*rand(n,1)-pi; % Random theta
phi = pi*rand(n,1) - pi/2; % Random phi
x = cos(phi).*cos(theta); % Create x values
y = cos(phi).*sin(theta); % Create y values
z = sin(phi); % Create z values
figure
plot3(x,y,z)

[k1,av1] = convhull(x,y,z,'Simplify',true);
hold on
h = trisurf(k1,x,y,z,'FaceColor','cyan')
h.FaceAlpha = .3
axis equal

xyz = 2*rand(3, n)-1; % Generate random points
tri = delaunayn([x y z]); % Generate delaunay triangulization
tn = tsearchn([x y z], tri, xyz'); % Determine which triangle point is within
IsInside = ~isnan(tn); % Convert to logical vector
for iii = 1 : length(IsInside)
    if IsInside(iii) == 1
        plot3(xyz(1,iii),xyz(2,iii),xyz(3,iii),'ob')
    else
        plot3(xyz(1,iii),xyz(2,iii),xyz(3,iii),'or')
    end
end



%%

%% step 9 - Stitching

dfmax = 4; % maximum number of tolerated missing frames to reconnect to trajectories
dxmax = 2*0.031; % (mm) % maximum tolerated distance between stitchs parts
dvmax = 0.3;
lmin  = 2*0.031;
StitchedTraj = Stitching(session,nameExpe,trackName,dfmax,dxmax,dvmax,lmin);



%%


%%


%% OLD STUFF



%%



%%
colorPlots = jet(2001);

%someTrajectories = struct();
%nTraj_gi = length(someTrajectories) + 1;
%nTraj_gi = 1;
% 1 . choose two trajectories
hcam0 = figure; hold on, box on
for ic = 1 : length(trajArray_CAM1)
    if length(trajArray_CAM1(ic).track) > 40
        clear xC1 yC1
        xC1 = [trajArray_CAM1(ic).track(:,1)];
        yC1 = [trajArray_CAM1(ic).track(:,2)];
        plot(xC1,yC1,'-ob')
    end
end
for ic = 1 : length(trajArray_CAM2)
    if length(trajArray_CAM2(ic).track) > 40
        clear xC1 yC1
        xC1 = [trajArray_CAM2(ic).track(:,1)];
        yC1 = [trajArray_CAM2(ic).track(:,2)];
        plot(xC1,yC1,'-or')
    end
end
set(gca,'ydir','reverse')
set(gcf,'position',[26   160   915   809])
title('cam01')


%%
colorPlots = jet(size(CC1,2)+1);

%someTrajectories = struct();
%nTraj_gi = length(someTrajectories) + 1;
%nTraj_gi = 1;
% 1 . choose two trajectories
hcam0 = figure; hold on, box on
for ic = 1 : length(trajArray_CAM1)
    if length(trajArray_CAM1(ic).track) > 40
        clear xC1 yC1
        xC1 = [trajArray_CAM1(ic).track(:,1)];
        yC1 = [trajArray_CAM1(ic).track(:,2)];
        plot(xC1,yC1,'-ob',...
            'markerEdgeColor','none',...
            'markerFaceColor',colorPlots(min(trajArray_CAM1(ic).track(:,3)),:))
    end
end
set(gca,'ydir','reverse')
set(gcf,'position',[26   160   915   809])
title('cam01')
xC1 = [trajArray_CAM1(itrajCam0).track(:,1)];
yC1 = [trajArray_CAM1(itrajCam0).track(:,2)];
plot(xC1,yC1,'-r','lineWidth',4)

hcam1 = figure; hold on, box on
for ic = 1 : length(trajArray_CAM2RAW)
    if length(trajArray_CAM2RAW(ic).track) > 40
        clear xC2 yC2
        xC2 = [trajArray_CAM2RAW(ic).track(:,1)];
        yC2 = [trajArray_CAM2RAW(ic).track(:,2)];
        plot(xC2,yC2,'-ob',...
            'markerEdgeColor','none',...
            'markerFaceColor',colorPlots(min(trajArray_CAM2RAW(ic).track(:,3)),:))
    end
end
set(gca,'ydir','reverse')
set(gcf,'position',[950   160   915   809])
title('cam02')
xC1 = [trajArray_CAM2RAW(itrajCam1).track(:,1)];
yC1 = [trajArray_CAM2RAW(itrajCam1).track(:,2)];
plot(xC1,yC1,'-r','lineWidth',4)
    


%% 000 - DEBUGGING ZONE: Exploration matching tracks


%% cross rays with trajectory selected one by one by hand
ci = clock; fprintf('start at %0.2dh%0.2dm\n',c(4),c(5))

someTrajectories = struct();

Ttype = 'T3';
calibTemp = load(CalibFile,'calib'); calib = calibTemp.calib;
CalibFileCam1 = calib(:,1);
CalibFileCam2 = calib(:,2);

for iselTraj = 1 : size(cam1cam2RAW,1)
    xt1 = cam1cam2RAW(iselTraj,1);
    yt1 = cam1cam2RAW(iselTraj,2);
    xt2 = cam1cam2RAW(iselTraj,3);
    yt2 = cam1cam2RAW(iselTraj,4);
    % find the trajectory in camera 01
    clear dgi
    dgi = nan(length(trajArray_CAM1),1);
    for ic = 1 : length(trajArray_CAM1)
        clear xC1 yC1 dd
        xC1 = [trajArray_CAM1(ic).track(:,1)];
        yC1 = [trajArray_CAM1(ic).track(:,2)];
        dd = sqrt((xC1-xt1).^2 + (yC1-yt1).^2);
        dgi(ic) = min(dd);
    end
    [~,itraj1] = min(dgi);
    figure
    plot([trajArray_CAM1(itraj1).track(:,1)],[trajArray_CAM1(itraj1).track(:,2)],'-ob')
    % find the trajectory in camera 02
    clear dgi
    dgi = nan(length(trajArray_CAM2RAW),1);
    for ic = 1 : length(trajArray_CAM2RAW)
        clear xC2 yC2 dd2
        xC2 = [trajArray_CAM2RAW(ic).track(:,1)];
        yC2 = [trajArray_CAM2RAW(ic).track(:,2)];
        dd2 = sqrt((xC2-xt2).^2 + (yC2-yt2).^2);
        dgi(ic) = min(dd2);
    end
    [~,itraj2] = min(dgi);
    
    figure
    plot([trajArray_CAM2RAW(itraj2).track(:,1)],[trajArray_CAM2RAW(itraj2).track(:,2)],'-ob')
    
    someTrajectories(iselTraj).itraj1 = itraj1;
    someTrajectories(iselTraj).itraj2 = itraj2;
    
    % cross the two choosen rays
    clear x01 y01 x02 y02 x02incam01 y02incam01
    tminCAM01 = min(trajArray_CAM1(itraj1).track(:,3));
    tmaxCAM01 = max(trajArray_CAM1(itraj1).track(:,3));
    tminCAM02 = min(trajArray_CAM2RAW(itraj2).track(:,3));
    tmaxCAM02 = max(trajArray_CAM2RAW(itraj2).track(:,3));
    [A,B,C] = intersect([tminCAM01:tmaxCAM01],[tminCAM02:tmaxCAM02]);
    
    A;
    if A
        clear x01 y01 x02 y02
        x01 = trajArray_CAM1(itraj1).track(min(B):max(B),1);
        y01 = trajArray_CAM1(itraj1).track(min(B):max(B),2);
        x02 = trajArray_CAM2RAW(itraj2).track(min(C):max(C),1);
        y02 = trajArray_CAM2RAW(itraj2).track(min(C):max(C),2);
        
        clear x_pxC1 y_pxC1 x_pxC2 y_pxC2
        for ixy = 1 : length(x01)
            
            x_pxC1 = x01(ixy);
            y_pxC1 = y01(ixy);
            x_pxC2 = x02(ixy);
            y_pxC2 = y02(ixy);
            
            [crossP,D] = crossRaysonFire(CalibFileCam2,CalibFileCam1,x_pxC1,y_pxC1,x_pxC2,y_pxC2,Ttype);
            if length(crossP)>0
                someTrajectories(iselTraj).x3D(ixy) = crossP(1);
                someTrajectories(iselTraj).y3D(ixy) = crossP(2);
                someTrajectories(iselTraj).z3D(ixy) = crossP(3);
                someTrajectories(iselTraj).D(ixy) = D;
            end
        end
    end
    
    
end
ce = clock; fprintf('done at %0.2dh%0.2dm in %0.0f s \n',c(4),c(5), etime(ce,ci))

%%
figure, hold on, box on
for iselTraj = 1 : size(cam1cam2RAW,1)
    xt1 = cam1cam2RAW(iselTraj,1);
    yt1 = cam1cam2RAW(iselTraj,2);
    xt2 = cam1cam2RAW(iselTraj,3);
    yt2 = cam1cam2RAW(iselTraj,4);
    % find the trajectory in camera 01
    clear dgi
    dgi = nan(length(trajArray_CAM1),1);
    for ic = 1 : length(trajArray_CAM1)
        clear xC1 yC1 dd
        xC1 = [trajArray_CAM1(ic).track(:,1)];
        yC1 = [trajArray_CAM1(ic).track(:,2)];
        dd = sqrt((xC1-xt1).^2 + (yC1-yt1).^2);
        dgi(ic) = min(dd);
    end
    [~,itraj1] = min(dgi);
    plot([trajArray_CAM1(itraj1).track(:,1)],[trajArray_CAM1(itraj1).track(:,2)],'-ob')
end
set(gca,'ydir','reverse')

hist3D = [];
% plot the result
figure, hold on
for itraj3D = 1 : size(cam1cam2RAW,1)
    itraj3D
    plot3([someTrajectories(itraj3D).x3D],...
        [someTrajectories(itraj3D).y3D],...
        [someTrajectories(itraj3D).z3D],'og')
     hist3D = [hist3D,[someTrajectories(itraj3D).z3D]]; 
end
view(3)
xlabel('x')
ylabel('y')
zlabel('z')
axis equal





%% STEP 6 - centers to rays - BLP
%  vérifier qu'il n'y a pas de bug ici !!!!
%  genre x et y inversés

tic
camPV = struct();
clear CC1_1P_timecat CC2_1P_timecat
CC1_1P_timecat   = struct();    CC2_1P_timecat   = struct();
CC1_1P_timecat.X = [];          CC2_1P_timecat.X = [];
CC1_1P_timecat.Y = [];          CC2_1P_timecat.Y = [];
for it = [10:30] % je combine ensemble des points de deux temps différents
    it
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %removing the NaNs
    CC1_1P = CC1;
    ikill = [];
    for ip = 1 : size(CC1_1P(it).X,2)
        if isnan(CC1_1P(it).X(ip)) || isnan(CC1_1P(it).Y(ip))
            ikill = [ikill,ip];
        end
    end
    CC1_1P(it).X(ikill) = [];
    CC1_1P(it).Y(ikill) = [];
    % keeping only one particle for testing the code
    xmin = 149;
    xmax = 156;
    ymin = 800;
    ymax = 812;
    %removing
    i2rmv_xmin = CC1_1P(it).X < xmin;
    i2rmv_xmax = CC1_1P(it).X > xmax;
    i2rmv_ymin = CC1_1P(it).Y < ymin;
    i2rmv_ymax = CC1_1P(it).Y > ymax;
    i2rmv = or(or(i2rmv_xmin,i2rmv_xmax),or(i2rmv_ymin,i2rmv_ymax));
    CC1_1P(it).X(i2rmv) = [];
    CC1_1P(it).Y(i2rmv) = [];
    %figure
    %plot(CC1_1P(it).X,CC1_1P(it).Y,'+r')
    
    CC2_1P = CC2;
    ikill = [];
    for ip = 1 : size(CC2_1P(it).X,2)
        if isnan(CC2_1P(it).X(ip)) || isnan(CC2_1P(it).Y(ip))
            ikill = [ikill,ip];
        end
    end
    CC2_1P(it).X(ikill) = [];
    CC2_1P(it).Y(ikill) = [];
    % keeping only one particle for testing the code
    xmin = 267;
    xmax = 273;
    ymin = 670;
    ymax = 680;
    %removing
    clear i2rmv_xmin i2rmv_xmax i2rmv_ymin i2rmv_ymax i2rmv
    i2rmv_xmin = CC2_1P(it).X < xmin;
    i2rmv_xmax = CC2_1P(it).X > xmax;
    i2rmv_ymin = CC2_1P(it).Y < ymin;
    i2rmv_ymax = CC2_1P(it).Y > ymax;
    i2rmv = or(or(i2rmv_xmin,i2rmv_xmax),or(i2rmv_ymin,i2rmv_ymax));
    CC2_1P(it).X(i2rmv) = [];
    CC2_1P(it).Y(i2rmv) = [];
    
    CC1_1P_timecat.X = [CC1_1P_timecat.X,CC1_1P(it).X]; CC1_1P_timecat.Y = [CC1_1P_timecat.Y,CC1_1P(it).Y];
    CC2_1P_timecat.X = [CC2_1P_timecat.X,CC2_1P(it).X]; CC2_1P_timecat.Y = [CC2_1P_timecat.Y,CC2_1P(it).Y];
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
for icam = 1:2
    clear ikill calib pos XYZ P V CC
    if icam == 1
        CC(it) = CC1_1P_timecat;
    elseif icam ==2
        CC(it) = CC2_1P_timecat;
    end
    ikill = [];
    calib = load(CalibFile);
    calib = calib.calib;
    calib = calib(:,icam);
    %     for ical = 1 : size(calib,1)
    %         calib(ical).dirPlane = [2,3,1];
    %     end
    pos(1:length(CC(it).X),1) = [CC(it).X]';
    pos(1:length(CC(it).Y),2) = [CC(it).Y]';
    for ip = 1 : size(pos(:,1))
        if isnan(pos(ip,1)) || isnan(pos(ip,2))
            ikill = [ikill,ip];
        end
    end
    pos(ikill,:) = [];
    
    [XYZ]=TAN_Tpx2rw(calib,pos);
    [P,V]=TAN_fit3Dline(XYZ);
    
    camPV(icam,it).P = P;
    camPV(icam,it).V = V;
end

toc



%% step 7 - sur le PSMN - soon on Matlab

cd 'D:\pono\IFPEN\IFPEN_manips\expe_2021_03_11\for4DPTV\re01_10spatules\Processed_DATA\expe_2021_03_11_re01_spatule10_zaber100mm'
filename = 'rays_out_cpp.h5';
info = h5info(filename);
%%
% data = h5read(filename,'/tracks')
%% step 7 - vizualisation of the matching

%% step 8 - Tracking

session.input_path = strcat('D:\pono\IFPEN\IFPEN_manips\expe_2021_03_11\for4DPTV\',...
    're01_10spatules\');
session.output_path = session.input_path;
trackName = strcat(nameExpe,'_rays_out');
[tracks,traj]=track3d(session, nameExpe, 'rays_out_cpp',10,0.2,2,1,2,1);

%%


cd 'D:\pono\IFPEN\IFPEN_manips\expe_2021_03_11\for4DPTV\re01_10spatules\Processed_DATA\expe_2021_03_11_re01_spatule10_zaber100mm'
filename = 'tracks_rays_out_cpp.h5';
info = h5info(filename);
L=h5read(filename,'/L');

datapath = fullfile(session.input_path,'Processed_DATA',nameExpe,filename(1:end-3));
traj = h52tracks(datapath);





%% Test on 2021 05 28 experiment
figure
plot(Camera0_FIJI(:,1),Camera0_FIJI(:,2),'-ob')
plot(Camera1_FIJI(:,1),Camera1_FIJI(:,2),'ob')

%%
FIJIrt= struct(); % fiji rapid test
ip3D = 0;
for ipcam0 = 1 : size(Camera0_FIJI,1)
    % find in the other camera if there is points at the same time
    listTestPoints = find(Camera0_FIJI(ipcam0,3) == Camera1_FIJI(:,3));
    for iOK = 1 : length(listTestPoints)
        x_pxC1 = Camera0_FIJI(ipcam0,1);
        y_pxC1 = Camera0_FIJI(ipcam0,2);
        x_pxC2 = Camera1_FIJI(listTestPoints(iOK),1);
        y_pxC2 = Camera1_FIJI(listTestPoints(iOK),1);
        clear crossP D
        [crossP,D] = crossRaysonFire(CalibFileCam1,CalibFileCam2,x_pxC1,y_pxC1,x_pxC2,y_pxC2,Ttype);
        if size(crossP,1)
            ip3D = ip3D + 1;
            FIJIrt.x3D(ip3D) = crossP(1);
            FIJIrt.y3D(ip3D) = crossP(2);
            FIJIrt.z3D(ip3D) = crossP(3);
            FIJIrt.t(ip3D)   = Camera0_FIJI(ipcam0,3);
            FIJIrt.D(ip3D)   = D;
        end
    end
end
%%
colT = jet(500);
figure, hold on
for ip = 1 : length([FIJIrt.x3D])
    plot3(FIJIrt.x3D(ip),FIJIrt.y3D(ip),FIJIrt.z3D(ip),'o',...
        'markerFaceColor',colT(FIJIrt.t(ip),:),...
        'markerEdgeColor',colT(FIJIrt.t(ip),:))
end
view(3)
box on
xlabel('x')
ylabel('y')
zlabel('z')
%% FUNCTIONS

%%
% weighted centroid or gaussian approximation

ifile = 35;

fprintf('load image sequence \n')
inputFolder = allExpeStrct(iexpe).inputFolder;
cd(inputFolder)

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
%fprintf('line 1339 \n')

%%

th = allExpeStrct(iexpe).centerFinding_th;
sz = allExpeStrct(iexpe).centerFinding_sz;
Nwidth = 1;
for it = 1 % : size(M,3)
    %fprintf('doing time %0.4d / %0.4d \n',it,size(M,3))
    CC(it).xyRAW = pkfnd(Im01(:,:,it),th,sz);
    
    tic
    for ixy = 1 : length(CC(it).xyRAW)
        % refine at subpixel precision
        Im = zeros(size(Im01,1),size(Im01,2),class(Im01));
        Im(:,:) = Im01(:,:,it);
        
        clear xpkfnd ypkfnd Ip
        Ip = zeros(2*Nwidth+1,2*Nwidth+1,'double');
        xpkfnd = CC(it).xyRAW(ixy,1);
        ypkfnd = CC(it).xyRAW(ixy,2);
        Ip = double(Im(ypkfnd-Nwidth:ypkfnd+Nwidth,xpkfnd-Nwidth:xpkfnd+Nwidth));
        CC(it).xy(ixy,1) = xpkfnd + 0.5*log(Ip(2,3)/Ip(2,1))/(log((Ip(2,2)*Ip(2,2))/(Ip(2,1)*Ip(2,3))));
        CC(it).xy(ixy,2) = ypkfnd + 0.5*log(Ip(3,2)/Ip(1,2))/(log((Ip(2,2)*Ip(2,2))/(Ip(1,2)*Ip(3,2))));
        
    end
    toc
    
    tic
    
    Im = zeros(size(Im01,1),size(Im01,2),class(Im01));
    Im(:,:) = Im01(:,:,it);
    stats = regionprops(Im>0,Im,'centroid','Area','weightedcentroid');
    ikill = find([stats.Area]<9);
    stats(ikill) = [];
    toc
    
end

figure
imagesc(Im), colormap gray
hold on
for  ixy = 1 : length(CC(it).xyRAW)
    plot(CC(it).xy(ixy,1),CC(it).xy(ixy,2),'or')
end
for istats = 1 : length(stats)
    xwc = stats(istats).WeightedCentroid(1);
    ywc = stats(istats).WeightedCentroid(2);
    plot(xwc,ywc,'+g')
end
caxis([0 10]);

figure
imagesc(M(:,:,it)), colormap gray
hold on
for  ixy = 1 : length(CC(it).xyRAW)
    plot(CC(it).xy(ixy,1),CC(it).xy(ixy,2),'or')
end
for istats = 1 : length(stats)
    xwc = stats(istats).WeightedCentroid(1);
    ywc = stats(istats).WeightedCentroid(2);
    plot(xwc,ywc,'+g')
end
caxis([0 10]);
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

lcrossStitchTHRSHLD = 4;
itStep = 4;
timeShift = 5;  % 5; % frames
rA = 20;
trajArray_CAM1 = Darcy02stitching(trajArray_CAM1,lcrossStitchTHRSHLD,itStep,timeShift,rA);

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


end

%% crossRaysonFire
function [crossP,D] = crossRaysonFire(CalibFileCam1,CalibFileCam2,x_pxC1,y_pxC1,x_pxC2,y_pxC2,Ttype)
D = 'nan';

[P1,V1]=findRaysDarcy02(CalibFileCam1,x_pxC1,y_pxC1,Ttype);
[P2,V2]=findRaysDarcy02(CalibFileCam2,x_pxC2,y_pxC2,Ttype);


if size(P1,1) == 0
    crossP = [];
elseif size(P2,1) == 0
    crossP = [];
else
    
    if size(P1,1) == 3
        P1 = P1';
    end
    if size(P2,1) == 3
        P2 = P2';
    end
    
    if isempty(P1)
        %break
    elseif isempty(P2)
        %break
    end
    
    
    clear lineA0 lineA1 lineB0 lineB1
    lineA0 = P1;
    lineA1 = (P1+V1);
    lineB0 = P2;
    lineB1 = (P2+V2);
    [D,Xcp,Ycp,Zcp,Xcq,Ycq,Zcq,Dmin,imin,jmin]= ll_dist3d(lineA0,lineA1,lineB0,lineB1);
    crossP = ([Xcp,Ycp,Zcp]+[Xcq,Ycq,Zcq])/2; % crossing oping
    
end

end

%% findRaysDarcy02
function [P,V,XYZ]=findRaysDarcy02(calib,x_px,y_px,Ttype)
%%% calib : calibration data for this camera
%%% x_px  : x coordinates in px,
%%% y_px  : y coordinates in px,
%%% Ttype : type of the transformation to use (T1=Linear, T3=Cubic).

% calibTemp = load(CalibFile,'calib'); calib = calibTemp.calib;

Npart = numel(x_px);
Nplans = numel(calib);

XYZ = zeros(numel(calib),3,numel(x_px));

for kplan = 1:Nplans
    I = inpolygon(x_px,y_px,calib(kplan).pimg(calib(kplan).cHull,1),calib(kplan).pimg(calib(kplan).cHull,2));
    if max(I)>0
        if Ttype=='T1'
            [Xtmp,Ytmp]=transformPointsInverse((calib(kplan).T1px2rw),x_px(I==1),y_px(I==1));
        elseif Ttype=='T3'
            [Xtmp,Ytmp]=transformPointsInverse((calib(kplan).T3px2rw),x_px(I==1),y_px(I==1));
        end
        
        XYZ(kplan,1,I==1)=Xtmp;
        XYZ(kplan,2,I==1)=Ytmp;
        XYZ(kplan,3,I==1)=calib(kplan).posPlane;
    end
    
    XYZ(kplan,1,I==0) = NaN;
    XYZ(kplan,2,I==0) = NaN;
    XYZ(kplan,3,I==0) = NaN;
end
[P, V] = fit3Dline(XYZ);


end

%% fit3Dline
function [xyz0,direction] = fit3Dline(XYZ)

if max(max(max(isnan(XYZ)))) ==0
    [xyz0,direction] = fit3Dline_nonan(XYZ);
else
    [P V] = arrayfun(@(I)(fit3Dline_nan(XYZ(:,:,I))),1:size(XYZ,3),'UniformOutput',false);
    xyz0 = (cell2mat(P'));
    direction = (cell2mat(V'));
    
    xyz0(isnan(xyz0)) = [];
    direction(isnan(direction)) = [];
end

end

%% TAN_fit3Dline
function [P,V]=TAN_fit3Dline(XYZ)
%%-------------------------------------------------------------------------
%%computes the line of best fit (in the least square sense) for points in
%%Three Dimensional Space using the 3D Orthogonal Distance Regression
%%(ODR) line method.
%%The line is parametrized by l = P + V*t, P(xo,yo,zo) is given by the mean
%%of all the points and V(u,v,w) by the eigenvector associated with the
%%largest singular value of the matrix M = [xi - xo , yi - yo , zi - zo]
%%XYZ is a 3D-matrix containing the set of points [xi yi zi] in the 2 first
%%dimensions, the third dimension correspond to each particle.
%%P is a 2D-matrix of the form P = [x0_1 y0_1 z0_1 ; ... ; x0_N y0_N z0_N]
%%V is a 2D-matrix of the form V = [u_1 v_1 w_1 ; ... ; u_N v_N w_N]
%%-------------------------------------------------------------------------

%XYZ(:,:,1) = [ 4 5 8 ; 7 5 9 ; 4 97 5 ; 78 6 13 ; 12 84 3];
%XYZ(:,:,2) = [ 78 75 58 ; 5 54 0 ; 74 97 50 ; 57 7 1 ; 1 4 8];
%XYZ(:,:,3) = [ 41 2 8 ; 7 10 9 ; 11 9 50 ; 78 60 103 ; 21 4 21];
%XYZ(:,:,4) = [ 13 4 0 ; 55 60 9 ; 1 92 2 ; 8 6 1 ; 2 4 1];

xyz0 = mean(XYZ,1);
P = squeeze(xyz0)';
M = XYZ - xyz0; %centering the data

[~, ~, Vec]=arrayfun(@(ii) svd(M(:,:,ii)),[1:size(M,3)],'UniformOutput',false);
Vac = cat(3,Vec{:});
V = squeeze(Vac(:,1,:))'; %in matlab the singular values are listed in decreasing order.

%dd=arrayfun(@(x) cross(Vac(:,end,x),Vac(:,end-1,x)),[1:size(Vac,3)],'UniformOutput',false);
%V=cat(2,dd{:})';  clear dd Vac A;

end

%% fit3Dline_nan
function [xyz0,direction]=fit3Dline_nan(XYZ)
%%% [xyz0,direction]=fit3Dline_jv(XYZ)
%
% @MBourgoin 01/2019

I = find(isnan(XYZ(:,1)));
XYZ(I,:)=[];

if size(XYZ,1)>2
    
    xyz0=mean(XYZ);
    %xyz0=cell2mat(arrayfun(@(x) mean(x.CCrw),Proj,'UniformOutput',false));
    
    A=bsxfun(@minus,XYZ,xyz0); %center the data
    
    % xyz0=XYZ(3,:);
    % A= XYZ;
    
    % xyz0=XYZ(plan_centre,:);
    % A=bsxfun(@minus,XYZ,xyz0); %center the data
    
    %[U,S,V]=svd(A);
    [Uac Sac Vac]=arrayfun(@(kkk) svd(A(:,:,kkk)),[1:size(A,3)],'UniformOutput',false);
    Ua=cat(3,Uac{:});
    Sa=cat(3,Sac{:});
    Va=cat(3,Vac{:}); clear Uac Sac Vac;
    
    %direction=cross(V(:,end),V(:,end-1));
    dd=arrayfun(@(x) cross(Va(:,end,x),Va(:,end-1,x)),[1:size(Va,3)],'UniformOutput',false);
    direction=cat(3,dd{:})';  clear dd;
else
    %xyz0 = [NaN NaN NaN];
    %direction = [NaN NaN NaN];
    xyz0=[];
    direction=[];
end

%line = [xyz0'  direction];
end

%% fit3Dline_nonan
function [xyz0,direction]=fit3Dline_nonan(XYZ)

% @JVessaire 01/2019

xyz0=mean(XYZ,1);
Aa=bsxfun(@minus,XYZ,xyz0); %center the data
xyz0=squeeze(xyz0)';

%Aa=permute(A,[3 2 1]);

[~, ~, Vac]=arrayfun(@(kkk) svd(Aa(:,:,kkk)),[1:size(Aa,3)],'UniformOutput',false);
Va=cat(3,Vac{:});

dd=arrayfun(@(x) cross(Va(:,end,x),Va(:,end-1,x)),[1:size(Va,3)],'UniformOutput',false);
direction=cat(2,dd{:})'; clear dd Vac A;

end

%% ll_dist3d
function [D,Xcp,Ycp,Zcp,Xcq,Ycq,Zcq,Dmin,imin,jmin]= ll_dist3d(P0,P1,Q0,Q1)
%ll_dist3d - Find the distances between each pair of straight 3D lines in
% two sets. Find the closest points on each pair, and the pair with minimum
% distance. Each line is defined by two distinct points.
%
% Input:
% P0 - array of first points of the first set (m X 3), where m is the
% number of lines in the first set. P0(j,1), P0(j,2), P0(j,3) are X, Y
% and X coordinates, accordingly, of point j.
% Pl - array of second points of the first set (m X 3), where m is the
% number of lines in the first set. P1(j,1), Pl(j,2), Pl(j,3) are X, Y
% and X coordinates, accordingly, of point j.
% Q0 - array of first points of the second set (n % 3), where n is the
% number of lines in the second set. Q0(k,1), Q0(k,2), Q0(k,3) are X, Y
% and X coordinates, accordingly, of point k.
% Ql - array of second points of the second set (n % 3), where n is the
% number of lines in the second set. Q0(k,1), Q0(k,2), Q0(k,3) are X, Y
% and X coordinates accordingly of point k.
% Output:
% D - array of distances between line pairs (m X n). D(j,k) is the
% distance between line j from the first (P) set, and line k from the
% second (Q) set.
% Xcp - array of X coordinates of closest points belonging to the first
% (P) set (m X n). Xcp(j,k) is an % coordinate of the closest point on a
% line j defined by P0(j,:) and P1(j,:), computed to the line k defined
% by Q0(k,:) and Q1(k,:).
% Ycp - array of Y coordinates of closest points belonging to the first
% (P) set (m X n). See Xcp definition.
% Zcp - array of Y coordinates of closest points belonging to the first
% (P) set (m X n). See Xcp definition.
% Xcq - array of X coordinates of closest points belonging to the second
% (Q) set (m X n). Xcq(j,k) is an % coordinate of the closest point on a
% line k defined by Q0(k,:) and Q1(k,:), computed to the line j defined
% by P0(j,:) and P1(1,:).
% Ycq - array of Y coordinates of closest points belonging to the second
% (Q) set (m X n). See Xcq definition.
% Zcq - array of % coordinates of closest points belonging to the second
% (Q) set (m X n). See Xcq definition.
%
% Remarks:
% Below is a simple unit test for this function. The test creates
% 2 sets of random 3D lines, finds the distances between each pair of
% lines, and plots the pair with shortest distance
% To run the test, uncommnent the following lines:
%
% n1 = 4; % number of lines in first set
% n2 = 2; % number of lines in first set
% P0 = rand(n1,3); P1 = rand(n1,3); Q0 = rand(n2,3); Q1 = rand(n2,3);
% [D,Xcp,Ycp,Zcp,Xcq,Ycq,Zcq,Dmin,imin,jmin] = ll_dist3d(P0, P1, Q0, Q1);
% t = (-2:0.01:2);
% Tp = repmat(t(:), 1, size(P0,1));
% Tq = repmat(t(:), 1, size(Q0,1));
% Xp = repmat(P0(:,1)',[size(t,2), 1]) + Tp.*(repmat(P1(:,1)',[size(t,2),1])-...
% repmat(P0(:,1)', size(t,2), 1));
% Yp = repmat(P0(:,2)',[size(t,2), 1]) + Tp.*(repmat(P1(:,2)',[size(t,2),1])-...
% repmat(P0(:,2)', size(t,2), 1));
% Zp = repmat(P0(:,3)',[size(t,2), 1]) + Tp.*(repmat(P1(:,3)',[size(t,2),1])-...
% repmat(P0(:,3)', size(t,2), 1));
% Xq = repmat(Q0(:,1)', size(t,2), 1) + Tq.*(repmat(Q1(:,1)',size(t,2),1)-...
% repmat(Q0(:,1)', size(t,2), 1));
% Yq = repmat(Q0(:,2)',size(t,2), 1) + Tq.*(repmat(Q1(:,2)',size(t,2),1)-...
% repmat(Q0(:,2)', size(t,2), 1));
% Zq = repmat(Q0(:,3)',size(t,2), 1) + Tq.*(repmat(Q1(:,3)',size(t,2),1)-...
% repmat(Q0(:,3)', size(t,2), 1));
% figure;
% plot3(Xp(:,imin),Yp(:,imin),Zp(:,imin),Xq(:,jmin),Yq(:,jmin),Zq(:,jmin));
% hold on
% plot3(Xcp(imin,jmin),Ycp(imin,jmin),Zcp(imin,jmin),'ro',Xcq(imin,jmin),Ycq(imin,jmin),Zcq(imin,jmin),'mo');
% axis equal
% grid on
% xlabel('X'); ylabel('Y'); zlabel('Z');
%
% Revision history:
% March 03, 2016 - created (Michael Yoshpe)
%**************************************************************************
% check inputs validity
[mp0, np0] = size(P0);
if(np0 ~=3 )
    error('Array P0 should of size (m X 3)');
end
[mpl, npl] = size(P1);
if((mpl ~= mp0) || (npl ~= np0))
    error('P0 and Pl arrays must be of same size');
end
[mq0, nq0] = size(Q0);
if(nq0 ~= 3)
    error('Array Q0 should of size (n X 3)');
end
[mq1, nq1] = size(Q1);
if((mq1 ~= mq0) || (nq1 ~= nq0))
    error('Q0 and Ql arrays must be of same size');
end
u = P1 - P0; % vectors from P0 to P1
uu = repmat(u,[1,1,mq0]);
v = Q1 - Q0; % vectors from Q0 to Q1
vv = permute(repmat(v,[1,1,mp0]), [3 2 1]);
PP0 = repmat(P0,[1,1,mq0]);
QQ0 = permute(repmat(Q0,[1,1,mp0]), [3 2 1]);
w0 = PP0 - QQ0;
aa = dot(uu,uu,2);
bb = dot(uu,vv,2);
cc = dot(vv,vv,2);
dd = dot(uu,w0,2);
ee = dot(vv,w0,2);
ff = aa.*cc - bb.*bb;
idx_par = (ff < 5*eps); % indices of parallel lines
idx_nonpar = ~idx_par; % indices of non-parallel lines
sc = NaN(mp0,1,mq0);
tc = NaN(mp0,1,mq0);
sc(idx_nonpar) = (bb(idx_nonpar).*ee(idx_nonpar) - ...
    cc(idx_nonpar).*dd(idx_nonpar))./ff(idx_nonpar);
tc(idx_nonpar) = (aa(idx_nonpar).*ee(idx_nonpar) - ...
    bb(idx_nonpar).*dd(idx_nonpar))./ff(idx_nonpar);
PPc = PP0 + repmat(sc, [1,3,1]).*uu;
QQc = QQ0 + repmat(tc, [1,3,1]).*vv;
Xcp = permute(PPc(:,1,:), [1 3 2]);
Ycp = permute(PPc(:,2,:), [1 3 2]);
Zcp = permute(PPc(:,3,:), [1 3 2]);
Xcq = permute(QQc(:,1,:), [1 3 2]);
Ycq = permute(QQc(:,2,:), [1 3 2]);
Zcq = permute(QQc(:,3,:), [1 3 2]);
% If there are parallel lines, find the distances  between them
% Note, that for parallel lines, the closest points will be undefined
% (will contain NaN's)
if(any(idx_par))
    idx_par3 = repmat(idx_par, [1,3,1]); % logical indices
    PPc(idx_par3) = PP0(idx_par3);
    tmpl = repmat(dd(idx_par)./bb(idx_par), [1, 3, 1]);
    tmp2 = vv(find(idx_par3));
    
    QQc(idx_par3) = QQ0(idx_par3) + tmpl(:).*tmp2;
end
PQc = (PPc - QQc);
D = permute(sqrt(dot(PQc,PQc,2)), [1 3 2]);
[Dmin, idx_min] = min(D(:));
[imin,jmin] = ind2sub(size(D), idx_min);
end

%% imageCorrelation
% debug the cleaning of C (varargin in function)
function [xoffSet,yoffSet] = imageCorrelation(xc,yc,ACC1,ACC2,w,filterOrder,varargin)
% varargin: ,'cleanC',dxPass01,dyPass01,R);
ACC1sub = zeros(w+1,w+1,'uint8');
ACC1sub = ACC1(yc-w:yc+w,xc-w:xc+w);
C = normxcorr2(imgaussfilt(ACC1sub,filterOrder),imgaussfilt(ACC2,filterOrder));


% set C to zero above a predefined radius
% Checking varargin structure
if ( length(varargin) > 1 )
    %fprintf('cleaning C \n')
    dxPass01 = double(varargin{:,2});
    dyPass01 = double(varargin{:,3});
    R = double(varargin{:,4});
    x0 = round(xc+dxPass01 + size(ACC1sub,1)/2);
    y0 = round(yc+dyPass01 + size(ACC1sub,2)/2);
    x = 1:size(C,2);
    y = 1:size(C,1);
    [xx,yy] = meshgrid(x,y);
    %     figure
    %     imagesc(C)
    C(((xx-x0).^2+(yy-y0).^2) > R^2)=0;
    %     figure
    %     imagesc(C)
end

%
[ypeak,xpeak] = find(C==max(C(:)));
yoffSet = ypeak-size(ACC1sub,1) + w;
xoffSet = xpeak-size(ACC1sub,2) + w;
end

%% DARCY02_matchingTracks

function [itraj2,dtraj,listPotTracks,prelist] = DARCY02_matchingTracks(itrajCam0,trajArray_CAM1,trajArray_CAM2RAW,tform1)
% We indicate the trajectory in camera 0,
% it finds the trajectory in camera 1
%
minTintersect = 10; % necessary overlapping time
distThresh = 10;
dstimestep = 5;

tminCAM01 = min(trajArray_CAM1(itrajCam0).track(:,3));
tmaxCAM01 = max(trajArray_CAM1(itrajCam0).track(:,3));
tmean = round(     length(trajArray_CAM1(itrajCam0).track(:,3))/2     );
xcam0 = trajArray_CAM1(itrajCam0).track(tmean,1);
ycam0 = trajArray_CAM1(itrajCam0).track(tmean,2);
[xcam1,ycam1] = transformPointsInverse(tform1,xcam0,ycam0);


% build the list of potential matching trajectory
% criteria for matching:
% 1/ having points at the same time
% 2/ being in the same region
dgi2cam1 = nan(1,length(trajArray_CAM2RAW));
listPotTracks = struct(); % list of potential tracks
ipt = 0; % i possible tracks

% a sort of prematching
for ic = 1 : length(trajArray_CAM2RAW)
    tmean = round( length(trajArray_CAM2RAW(ic).track(:,3))/2 );
    xcam1ic = trajArray_CAM2RAW(ic).track(tmean,1);
    ycam1ic = trajArray_CAM2RAW(ic).track(tmean,2);
    dtraj(ic) = sqrt((xcam1-xcam1ic)^2 + (ycam1-ycam1ic)^2);
end
prelist = find(dtraj<distThresh);

for iic = 1 : length(prelist) %1 : length(trajArray_CAM2RAW)
   ic = prelist(iic);
    tminCAM02 = min(trajArray_CAM2RAW(ic).track(:,3));
    tmaxCAM02 = max(trajArray_CAM2RAW(ic).track(:,3));
    clear A B C
    [A,tcam01,tcam02] = intersect([tminCAM01:tmaxCAM01],[tminCAM02:tmaxCAM02]);

    if length(A) > minTintersect % 1/ having points at the same time
        % 2/ being in the same region
        clear xC2 yC2 dd
        xC2 = [trajArray_CAM2RAW(ic).track(:,1)];
        yC2 = [trajArray_CAM2RAW(ic).track(:,2)];
        dd = sqrt((xC2-xcam1).^2 + (yC2-ycam1).^2);
        dgi2cam1(ic) = min(dd);
        if min(dd) < distThresh
            ipt = ipt + 1;
            listPotTracks(ipt).itr = ic;
            
            clear dsCam0 dsCam1
            dsCam0 = 0; dsCam1 = 0;
            for ids = 1 : dstimestep : length(A)-dstimestep
                xc0i = trajArray_CAM1(itrajCam0).track(tcam01(ids),1);
                yc0i = trajArray_CAM1(itrajCam0).track(tcam01(ids),2);
                xc0f = trajArray_CAM1(itrajCam0).track(tcam01(ids+dstimestep),1);
                yc0f = trajArray_CAM1(itrajCam0).track(tcam01(ids+dstimestep),2);
                xc1i = trajArray_CAM2RAW(ic).track(tcam02(ids),1);
                yc1i = trajArray_CAM2RAW(ic).track(tcam02(ids),2);
                xc1f = trajArray_CAM2RAW(ic).track(tcam02(ids+dstimestep),1);
                yc1f = trajArray_CAM2RAW(ic).track(tcam02(ids+dstimestep),2);
                dsCam0 = dsCam0 + sqrt((xc0i-xc0f)^2+(yc0i-yc0f)^2);
                dsCam1 = dsCam1 + sqrt((xc1i-xc1f)^2+(yc1i-yc1f)^2);
            end
            listPotTracks(ipt).dsCam0 = dsCam0;
            listPotTracks(ipt).dsCam1 = dsCam1;
            
            listPotTracks(ipt).dsRatio = dsCam0 / dsCam1;
            listPotTracks(ipt).distance = min(dd);
        end
    end
end

itraj2 = [];
if ipt > 0
    [~,iitraj2] = min([listPotTracks.distance]);
    if listPotTracks(iitraj2).dsRatio < 1.5 && listPotTracks(iitraj2).dsRatio > 0.5
        itraj2 = listPotTracks(iitraj2).itr;
    else
        itraj2 = [];
    end
end

end


%% for orthoslice viewing
function allevents(src,evt)
evname = evt.EventName;
    switch(evname)
        case{'CrosshairMoved'}
            disp(['Crosshair moved previous position: ' mat2str(evt.PreviousPosition)]);
            disp(['Crosshair moved current position: ' mat2str(evt.CurrentPosition)]);
        case{'CrosshairMoving'}
            disp(['Crosshair moving previous position: ' mat2str(evt.PreviousPosition)]);
            disp(['Crosshair moving current position: ' mat2str(evt.CurrentPosition)]);
    end
end
 
%% function: calculate perpendicular to a vector in 3D

function v_perp = find_perp(v_input)
%FIND_PERP Finds one of the infinitely number of perpendicular vectors of
%   the input. The input vector v_input is a size 3,1 or 3,1 vector (only
%   3-dim supported)
    
    if length(v_input) ~= 3     % Can't be wrong dim if len=3
        error('find_perp:WrongSize','Input vector has wrong size');
    end
    
    if sum(v_input ~= 0) == 0
        error('find_perp:GivenZeroVector','Zero vector given as input');
    end
    
    v_perp = zeros(size(v_input));
    
    if sum(v_input ~= 0) == 3       % Every element is not-zero
        v_perp(1) = 1;
        v_perp(3) = -v_input(1)/(v_input(3));
    elseif sum(v_input ~= 0) == 2
        if v_input(1) == 0
            v_perp(1:2) = [1 1];
            v_perp(3) = -v_input(2)/(v_input(3));
        elseif v_input(2) == 0
            v_perp(1:2) = [1 1];
            v_perp(3) = -v_input(1)/(v_input(3));
        else
            v_perp(2:3) = [1 1];
            v_perp(1) = -v_input(2)/(v_input(1));
        end
    else
        if v_input(1) ~= 0
            v_perp(2) = 1;
        else
            v_perp(1) = 1;
        end
    end
    
    if abs(dot(v_perp,v_input)) > 1E-09      % Must take round-off into account (the dot product is not always perfect zero)
        error('find_perp:DotProdNotZero',...
            'A perp vector could not be found (failed dot product test). Might there be a bug?');
    end
end

%% Stitching

%% STITCHING 

function trajArray_CAM1_sttchd = Darcy02stitching(trajArray_CAM1,lcrossStitchTHRSHLD,itStep,timeShift,rA)
% lcrossStitchTHRSHLD = 4;
% itStep = 4;
% timeShift = 5;  % 5; % frames
% rA = 20;
%
% trajArray_CAM1_sttchd stitched trajectories

%%% Part 01 simplify tracks
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
    
    for iittime = 1 : length(timeSubSample) 
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
%%%% %%%% %%%%
%%%% %%%% %%%% calculate ds END
%%%% %%%% %%%%



%%% Part 02 stitch
continue2stich = 'on';
conversionsSTR = struct();
icSTR = 0;
while strcmp(continue2stich,'on') % as long as we can stitch we continue to stitch
    itA = 0;
    conversions = 0 ;
    while(1)% trouver une trach qui peut stitcher
        itA = itA+1;
        if itA > size(trajArray_CAM1,2)
        %fprintf('stitched %0.0f trajectories \n',conversions)
            if conversions == 0
                continue2stich = 'off';
            end
            break
        end

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
               
                Dt = (trajArray_CAM1(itB).smplTrack(2,3)-trajArray_CAM1(itB).smplTrack(1,3));
                Dx = (trajArray_CAM1(itB).smplTrack(1,1)-trajArray_CAM1(itB).smplTrack(2,1));
                Dy = (trajArray_CAM1(itB).smplTrack(1,2)-trajArray_CAM1(itB).smplTrack(2,2));
                
                vBsmplX = Dx/Dt;
                vBsmplY = Dy/Dt;
                xB_extra = trajArray_CAM1(itB).smplTrack(1,1) + ...
                    (- tA2B + trajArray_CAM1(itB).smplTrack(1,3)) * vBsmplX;
                yB_extra = trajArray_CAM1(itB).smplTrack(1,2) + ...
                    (- tA2B + trajArray_CAM1(itB).smplTrack(1,3)) * vBsmplY;
                
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

trajArray_CAM1_sttchd = trajArray_CAM1;

end

%% distance point to line,
%  from 

function d = point_to_line(pt, v1, v2)
      a = v1 - v2;
      b = pt - v2;
      d = norm(cross(a,b)) / norm(a);
end