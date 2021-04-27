%% ROBUST ESTIMATION PART 01 - load CC1 and CC2

% normxcorr2(template,A)
% normcxcorr finds where imageA is inside imageB. In our case, we select
% a part of the image from cameraA and where it corresponds in the other
% cam and find the corresponding section in camB.

% fitgeotrans -> transformation type: 'affine'
% this is the function that will give us the lineal transformation that
% transforms the points between cameras A and B.
% variables:
% moving points: x- and y-coordinates of control points in the image you
% want to transform
% fixedPoints — x- and y-coordinates of control points in the fixed image

% list of functions

close all
clear all

CalibFile = strcat('D:\IFPEN\IFPEN_manips\expe_2021_04_22_calibration\for4DPTV\calib.mat');
% CalibFile = strcat('D:\IFPEN\IFPEN_manips\expe_2021_04_22_calibration\for4DPTV\calib.mat');

him = 1152;
wim = 1152;
% load CC:
% cd('D:\IFPEN\IFPEN_manips\expe_2021_03_11\for4DPTV\re01_20spatules\Processed_DATA\zaber_100mm_20spatules_16bit_20210311T153131')

%cd('C:\Users\Lenovo\Desktop\IFPEN\DL\for4DPTV\Processed_DATA\visu01_20210402T160947')
%cd('C:\Users\Lenovo\Desktop\IFPEN\DL\for4DPTV\Processed_DATA\visu01_20210402T155814')
cd('D:\IFPEN\IFPEN_manips\expe_2021_04_20_beads\for4DPTV\Processed_DATA\expe65_20210420T172713')
%cd('C:\Users\Lenovo\Desktop\manip_20210420\expe65')
CCtemp = load('centers_cam1.mat', 'CC');
CC1 = CCtemp.CC;
CCtemp = load('centers_cam2.mat', 'CC');
CC2 = CCtemp.CC;
totalnFrames = size(CC1,2);

CC1(501:end) = [];
CC2(501:end) = [];
totalnFrames = size(CC1,2);

%% ROBUST ESTIMATION PART 1.1 removing the NaNs for all t
for it = 1 : size(CC1,2)
    ikill = [];
    for ip = 1 : size(CC1(it).X,2)
        if isnan(CC1(it).X(ip)) || isnan(CC1(it).Y(ip))
            ikill = [ikill,ip];
        end
    end
    CC1(it).X(ikill) = [];
    CC1(it).Y(ikill) = [];
    clear ikill
    ikill = [];
    for ip = 1 : size(CC2(it).X,2)
        if isnan(CC2(it).X(ip)) || isnan(CC2(it).Y(ip))
            ikill = [ikill,ip];
        end
    end
    CC2(it).X(ikill) = [];
    CC2(it).Y(ikill) = [];
end
%% ROBUST ESTIMATION PART 1.2 normxcorr2 - we build the images

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
%% ROBUST ESTIMATION PART 1.3 normxcorr2 pass 01 (on a large window) 
%  -- adapted for visu01_20210402T160947

close all
monPos = get(0,'MonitorPositions');
wmon=monPos(3); hmon=monPos(4);
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
xm = 350+round(wim/2);
ym = -50+round(him/2);
wsub = round(0.15*mean(xm,ym)); % width correlation template image
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

%% ROBUST ESTIMATION PART 1.3 normxcorr2 pass 01 (on a large window) 

close all
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
wsub = 250%round(0.25*mean(xm,ym)); % width correlation template image
[xoffSet,yoffSet] = imageCorrelation(xm,ym,ACC1,ACC2,wsub,filterOrder);

figure(hcam01), hold on
drawrectangle(gca,'Position',[xm-wsub,ym-wsub,2*wsub,2*wsub], ...
    'FaceAlpha',0,'Color','b');
figure(hcam02), hold on
drawrectangle(gca,'Position',[xoffSet-wsub,yoffSet-wsub,2*wsub,2*wsub], ...
    'FaceAlpha',0,'Color','r');
dxPass01 =   xoffSet-xm
dyPass01 =   yoffSet-ym
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

%%  ROBUST ESTIMATION PART 1.3 normxcorr2 pass 02 (on a small window)

wti = 250; % width template images
wstep = 100; % step for sampling the image
nPartMin = 200; % minimum number of particles to calculate the correlation
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

sprintf('done')

%% ROBUST ESTIMATION PART 1.4 build tform1
%fixedPoints = [163 427 963 951; 570 781 322 738]';
%movingPoints = [271 522 1002 1002; 456 631 198 573]';
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
%%
falseColorOverlay = imfuse( 1*ACC1, 1*ACC2);
imshow( falseColorOverlay, 'initialMagnification', 'fit');
set(gcf,'position',[ 189         122        1058         858])
%% ROBUST ESTIMATION PART 2.1 track particles in 2D on each camera

%% ROBUST ESTIMATION PART 2.1 from BLP TRAJECTOIRE 2D
tic
for it = 1 : size(CC1,2)
    part_cam1(it).pos(:,1) = [CC1(it).X]; % out_CAM1(:,1);
    part_cam1(it).pos(:,2) = [CC1(it).Y]; % out_CAM1(:,2);
    part_cam1(it).pos(:,3) = ones(length([CC1(it).X]),1)*it;
    part_cam1(it).intensity = 0; %mI;
    
    clear cam2X cam2Y
    [cam2X,cam2Y] = transformPointsForward(tform1,CC2(it).X,CC2(it).Y);
    part_cam2(it).pos(:,1) = [cam2X]; % out_CAM1(:,1);
    part_cam2(it).pos(:,2) = [cam2Y]; % out_CAM1(:,2);
    part_cam2(it).pos(:,3) = ones(length([cam2X]),1)*it;
    part_cam2(it).intensity = 0; %mI;
end
toc

tic
maxdist = 3;
longmin = 5;
[trajArray_CAM1,tracks_CAM1]=TAN_track2d(part_cam1,maxdist,longmin);

[trajArray_CAM2,tracks_CAM2]=TAN_track2d(part_cam2,maxdist,longmin);
toc

sprintf('done')
%% keep only long trajectories 
clear ikill01 ikill02
for itraj1 = 1 : length(trajArray_CAM1)
    lt1(itraj1) = size(trajArray_CAM1(itraj1).track,1) ; 
end
figure
histogram(lt1,[0:2:1000])
hold on
plot(10*[1 1],[0 400],'--k')
ikill01 = find(lt1<10);

for itraj2 = 1 : length(trajArray_CAM2)
    lt2(itraj2) = size(trajArray_CAM2(itraj2).track,1) ; 
end
figure
histogram(lt2)
ikill02 = find(lt2<10);

trajArray_CAM1(ikill01)=[];
trajArray_CAM2(ikill02)=[];

%% try to associate trajectories: prematch
c = clock; fprintf('start at %0.2dh%0.2dm\n',c(4),c(5))

% figure('defaultAxesFontSize',20), hold on, box on
structPotentialPairs = struct(); % structure listing potential pairs of trajectories
ipairs = 0;
tic

clear listPotTraj
% listPotTraj = struct(); % prematch, list of potential trajectories
tic
% idea to speed up: ne tester que les trajectoires qui sont prochent, avec
% une distance minimale dmin = 50 pix par exemple.
% prematch time is 0.2 sec
for itrj01 =  1 : length(trajArray_CAM1)
    clear xtCAM01 ytCAM01
    xtCAM01 = trajArray_CAM1(itrj01).track(:,1);
    ytCAM01 = trajArray_CAM1(itrj01).track(:,2);
    
    trajArray_CAM1(itrj01).L   = sqrt( (max(xtCAM01)-min(xtCAM01))^2 + (max(ytCAM01)-min(ytCAM01))^2);
    trajArray_CAM1(itrj01).x0 = mean(xtCAM01);
    trajArray_CAM1(itrj01).y0 = mean(ytCAM01);
end

for itrj02 =  1 :  length(trajArray_CAM2)
    clear xtCAM02 ytCAM02
    xtCAM02 = trajArray_CAM2(itrj02).track(:,1);
    ytCAM02 = trajArray_CAM2(itrj02).track(:,2);
    
    trajArray_CAM2(itrj02).L   = sqrt( (max(xtCAM02)-min(xtCAM02))^2 + (max(ytCAM02)-min(ytCAM02))^2);
    trajArray_CAM2(itrj02).x0 = mean(xtCAM02);
    trajArray_CAM2(itrj02).y0 = mean(ytCAM02);
    
end

clear Lcam01 Lcam02 Lcam01rm Lcam02rm LLcam0102
Lcam01 = [trajArray_CAM1.L]';
Lcam02 = [trajArray_CAM2.L];
Lcam01rm = repmat(Lcam01,1,length(trajArray_CAM2));
Lcam02rm = repmat(Lcam02,length(trajArray_CAM1),1);
LLcam0102 = Lcam01rm + Lcam02rm;

listPotTraj = - LLcam0102 + pdist2([[trajArray_CAM1.x0]',[trajArray_CAM1.y0]'],[[trajArray_CAM2.x0]',[trajArray_CAM2.y0]']);
toc

% coupler avec la liste des temps possibles
%%
figure
histogram(listPotTraj(:))
%% sans prematch
tic
for itrj01 = 1 : length(trajArray_CAM1)
    fprintf('progress: %0.4d / %0.4d \n',itrj01,length(trajArray_CAM1))
    clear xtCAM01 ytCAM01
    xtCAM01 = trajArray_CAM1(itrj01).track(:,1);
    ytCAM01 = trajArray_CAM1(itrj01).track(:,2);
    %axis([min(xtCAM01) max(xtCAM01) min(ytCAM01) max(ytCAM01)])
    tminCAM01 = min(trajArray_CAM1(itrj01).track(:,3));
    tmaxCAM01 = max(trajArray_CAM1(itrj01).track(:,3));
    
    % for itrj = 1 : length(trajArray_CAM2)
    %     % camera 2
    %     clear xt yt
    %     xt = trajArray_CAM2(itrj).track(:,1);
    %     yt = trajArray_CAM2(itrj).track(:,2);
    %     plot(xt,yt,'or','lineWidth',2)
    % end
    for itrj = 1 : length(trajArray_CAM2)
        tminCAM02 = min(trajArray_CAM2(itrj).track(:,3));
        tmaxCAM02 = max(trajArray_CAM2(itrj).track(:,3));
        [A,B,C] = intersect([tminCAM01:tmaxCAM01],[tminCAM02:tmaxCAM02]);
        if A
            % camera 2
            clear xt yt
            xt = trajArray_CAM2(itrj).track(:,1);
            yt = trajArray_CAM2(itrj).track(:,2);
            [xA,xB,xC] = intersect([round(min(xtCAM01)):round(max(xtCAM01))],[round(min(xt)):round(max(xt))]);
            if xA
                [yA,yB,yC] = intersect([round(min(ytCAM01)):round(max(ytCAM01))],[round(min(yt)):round(max(yt))]);
                if yA
                    plot(xtCAM01,ytCAM01,'-b','lineWidth',2)
                    plot(xt,yt,'-r','lineWidth',2)
                    ipairs = ipairs + 1;
                    structPotentialPairs(ipairs).trajCAM01 = itrj01;
                    structPotentialPairs(ipairs).trajCAM02 = itrj;
                    structPotentialPairs(ipairs).txA = A;
                    structPotentialPairs(ipairs).tCAM01 = B;
                    structPotentialPairs(ipairs).tCAM02 = C;
                end
            end
        end
    end
    %axis([min(xtCAM01) max(xtCAM01) min(ytCAM01) max(ytCAM01)])
end
toc

%% avec prematch

c = clock; fprintf('start associating trajectories at %0.2dh%0.2dm\n',c(4),c(5))

ipairs = 0;
clear structPotentialPairs
tic
figure('defaultAxesFontSize',20), box on, hold on
for itrj01 = 1 : length(trajArray_CAM1)
    % fprintf('progress: %0.4d / %0.4d \n',itrj01,length(trajArray_CAM1))
    
    clear xtCAM01 ytCAM01
    
    xtCAM01 = trajArray_CAM1(itrj01).track(:,1);
    ytCAM01 = trajArray_CAM1(itrj01).track(:,2);
    
    tminCAM01 = min(trajArray_CAM1(itrj01).track(:,3));
    tmaxCAM01 = max(trajArray_CAM1(itrj01).track(:,3));
    
    for itrj = 1 : length(trajArray_CAM2)
        
        if listPotTraj(itrj01,itrj) < 0
        tminCAM02 = min(trajArray_CAM2(itrj).track(:,3));
        tmaxCAM02 = max(trajArray_CAM2(itrj).track(:,3));
        [A,B,C] = intersect([tminCAM01:tmaxCAM01],[tminCAM02:tmaxCAM02]);
        if A
            % camera 2
            clear xt yt
            xt = trajArray_CAM2(itrj).track(:,1);
            yt = trajArray_CAM2(itrj).track(:,2);
            
            plot(xtCAM01,ytCAM01,'-b','lineWidth',2)
            plot(xt,yt,'-r','lineWidth',2)
            ipairs = ipairs + 1;
            structPotentialPairs(ipairs).trajCAM01 = itrj01;
            structPotentialPairs(ipairs).trajCAM02 = itrj;
            structPotentialPairs(ipairs).txA = A;
            structPotentialPairs(ipairs).tCAM01 = B;
            structPotentialPairs(ipairs).tCAM02 = C;
        end
        end
    end
    %axis([min(xtCAM01) max(xtCAM01) min(ytCAM01) max(ytCAM01)])
end
toc
axis([0 1152 0 1152])
c = clock; fprintf('done associating trajectories at %0.2dh%0.2dm\n',c(4),c(5))
%%
c = clock; fprintf('start at %0.2dh%0.2dm\n',c(4),c(5))
tic
% figure('defaultAxesFontSize',20), box on, hold on
%%%%%%%%%%
for iP = 1 : length(structPotentialPairs)
    % iP
    itrj01 = structPotentialPairs(iP).trajCAM01;
    itrj02 = structPotentialPairs(iP).trajCAM02;
    
    
    clear x01 y01 x02 y02 db
    x01 = trajArray_CAM1(itrj01).track(:,1);
    y01 = trajArray_CAM1(itrj01).track(:,2);
    x02 = trajArray_CAM2(itrj02).track(:,1);
    y02 = trajArray_CAM2(itrj02).track(:,2);
    
    
    for it = 1 : length([structPotentialPairs(iP).tCAM01])
        it1 = structPotentialPairs(iP).tCAM01(it);
        it2 = structPotentialPairs(iP).tCAM02(it);
        %plot([x01(it1),x02(it2)],[y01(it1),y02(it2)],'-g')
        db(it) = ((x01(it1)-x02(it2))^2 + (y01(it1)-y02(it2))^2)^(1/2);
    end
    
        it1tt = [structPotentialPairs(iP).tCAM01];
    it2tt = [structPotentialPairs(iP).tCAM02];
    
    Dx01 = x01(it1tt(end))-x01(it1tt(1));
    Dx02 = x02(it2tt(end))-x02(it2tt(1));
    Dy01 = y01(it1tt(end))-y01(it1tt(1));
    Dy02 = y02(it2tt(end))-y02(it2tt(1));
    
    structPotentialPairs(iP).Dx01 = Dx01;
    structPotentialPairs(iP).Dx02 = Dx02;
    structPotentialPairs(iP).Dy01 = Dy01;
    structPotentialPairs(iP).Dy02 = Dy02;
    structPotentialPairs(iP).d = mean(db);
end

c = clock; fprintf('done at %0.2dh%0.2dm\n',c(4),c(5))

%% good and bad pairing
tic
c = clock; fprintf('start at %0.2dh%0.2dm\n',c(4),c(5))
figure('defaultAxesFontSize',20), box on, hold on
for itrj01 = 1 : length(trajArray_CAM1)
    clear xtCAM01 ytCAM01
    xtCAM01 = trajArray_CAM1(itrj01).track(:,1);
    ytCAM01 = trajArray_CAM1(itrj01).track(:,2);
    plot(xtCAM01,ytCAM01,'-','Color', [1 0 0 .25],'lineWidth',1)
end
for itrj02 = 1 : length(trajArray_CAM2)
    clear xtCAM02 ytCAM02
    xtCAM02 = trajArray_CAM2(itrj02).track(:,1);
    ytCAM02 = trajArray_CAM2(itrj02).track(:,2);
    plot(xtCAM02,ytCAM02,'-','Color', [0 0 1 .25],'lineWidth',2)
end
% show pairs one by one
for iP = 1 : length(structPotentialPairs) % [69,102,108,114,120]

itrj01 = structPotentialPairs(iP).trajCAM01;
itrj02 = structPotentialPairs(iP).trajCAM02;

clear x01 y01 x02 y02
x01 = trajArray_CAM1(itrj01).track(:,1);
y01 = trajArray_CAM1(itrj01).track(:,2);
x02 = trajArray_CAM2(itrj02).track(:,1);
y02 = trajArray_CAM2(itrj02).track(:,2);

it1tt = [structPotentialPairs(iP).tCAM01];
it2tt = [structPotentialPairs(iP).tCAM02];


    Dx01 = structPotentialPairs(iP).Dx01;
    Dx02 = structPotentialPairs(iP).Dx02;
    Dy01 = structPotentialPairs(iP).Dy01;
    Dy02 = structPotentialPairs(iP).Dy02;
    ddd  = structPotentialPairs(iP).d;
    
if ddd < 20 && (abs(Dx01-Dx02)<5) && (abs(Dy01-Dy02)<5)
    plot(x01,y01,'-b','lineWidth',2)
    plot(x02,y02,'-r','lineWidth',2)
    structPotentialPairs(iP).matched = 1;
    for it = 1 : length([structPotentialPairs(iP).tCAM01])
        it1 = structPotentialPairs(iP).tCAM01(it);
        it2 = structPotentialPairs(iP).tCAM02(it);
        plot([x01(it1),x02(it2)],[y01(it1),y02(it2)],'-g')
    end
else
    
    structPotentialPairs(iP).matched = 0;
end
%title(sprintf('iP: %0.0f, d: %0.0f, Dx01: %0.0f, Dx02: %0.0f, Dy01: %0.0f, Dy02: %0.0f ',...
%    iP,structPotentialPairs(iP).d,Dx01, Dx02, Dy01, Dy02))
end
toc
ax = gca;
ax.XLim = [480  725];
ax.YLim = [185  510];
set(gcf,'position',[680   117   990   861])
c = clock; fprintf('done at %0.2dh%0.2dm\n',c(4),c(5))

%% Crossing the rays
c = clock; fprintf('on croise les doigts at %0.2dh%0.2dm\n',c(4),c(5))

Ttype = 'T1';
tic
%h3D = figure('defaultAxesFontSize',20); box on, hold on


for iP = 1 : 1 : length(structPotentialPairs) 
    fprintf('progress: %0.0f / %0.0f',iP,length(structPotentialPairs) )
if structPotentialPairs(iP).matched == 1
itrj01 = structPotentialPairs(iP).trajCAM01;
itrj02 = structPotentialPairs(iP).trajCAM02;

clear x01 y01 x02 y02
x01 = trajArray_CAM1(itrj01).track(structPotentialPairs(iP).tCAM01,1);
y01 = trajArray_CAM1(itrj01).track(structPotentialPairs(iP).tCAM01,2);
x02incam01 = trajArray_CAM2(itrj02).track(structPotentialPairs(iP).tCAM02,1);
y02incam01 = trajArray_CAM2(itrj02).track(structPotentialPairs(iP).tCAM02,2);
[ x02, y02] = transformPointsInverse(tform1,x02incam01,y02incam01);


for ixy = 1 : length(x01)
    
    x_pxC1 = x01(ixy);
    y_pxC1 = y01(ixy);
    x_pxC2 = x02(ixy);
    y_pxC2 = y02(ixy);
    clear P1 V1 P2 V2
    [P1,V1]=findRaysDarcy02(CalibFile,x_pxC1,y_pxC1,Ttype);
    [P2,V2]=findRaysDarcy02(CalibFile,x_pxC2,y_pxC2,Ttype);
    
    if size(P1,1) == 3
        P1 = P1';
    end
    if size(P2,1) == 3
        P2 = P2';
    end
    
    if isempty(P1)
        break
    elseif isempty(P2)
        break
    end
    %lVBW = 1000; % length rays backward
    %lVFW = 1000; % length rays frontward
    
    %plot3(P1(1)+V1(1)*[-lVBW lVFW],P1(2)+V1(2)*[-lVBW lVFW],P1(3)+V1(3)*[-lVBW lVFW],'b-')
    %plot3(P2(1)+V2(1)*[-lVBW lVFW],P2(2)+V2(2)*[-lVBW lVFW],P2(3)+V2(3)*[-lVBW lVFW],'r-')
    %view(3)
    
    %closest point:
    clear lineA0 lineA1 lineB0 lineB1
    lineA0 = P1;
    lineA1 = (P1+V1);
    lineB0 = P2;
    lineB1 = (P2+V2);
    [D,Xcp,Ycp,Zcp,Xcq,Ycq,Zcq,Dmin,imin,jmin]= ll_dist3d(lineA0,lineA1,lineB0,lineB1);
    crossP = ([Xcp,Ycp,Zcp]+[Xcq,Ycq,Zcq])/2; % crossing oping
    % figure(h3D), hold on
    % plot3(crossP(1),crossP(2),crossP(3),'og')
    structPotentialPairs(iP).x3D(ixy) = crossP(1);
    structPotentialPairs(iP).y3D(ixy) = crossP(2);
    structPotentialPairs(iP).z3D(ixy) = crossP(3);
end
%axis([crossP(1)-10 crossP(1)+10 crossP(2)-10 crossP(2)+10 crossP(3)-10 crossP(3)+10])
end
end
%axis equal
toc
c = clock; fprintf('rays crossed at %0.2dh%0.2dm\n',c(4),c(5))
%%
h3D = figure('defaultAxesFontSize',20); box on, hold on
view(3)
histx3D = [];
histy3D = [];
histz3D = [];
for iP = 1 : length(structPotentialPairs)
    if ~isempty(structPotentialPairs(iP).x3D)
        clear x3D y3D z3D
        x3D = structPotentialPairs(iP).x3D;
        y3D = structPotentialPairs(iP).y3D;
        z3D = structPotentialPairs(iP).z3D;
        plot3(x3D,y3D,z3D,'-b')
        
        histx3D = [histx3D,x3D];
        histy3D = [histy3D,y3D];
        histz3D = [histz3D,z3D];
    end
end
xlabel('x')
ylabel('y')
zlabel('z')
axis([-300 300 -20 130 17 24])

figure
histogram(histx3D)
title('x')
figure
histogram(histy3D)
title('y')
figure
histogram(histz3D,[17:0.1:24])
title('z')
pause(2)
%%
 clear P1 V1 P2 V2
    [P1,V1]=findRaysDarcy02(CalibFile,x_pxC1,y_pxC1,Ttype)
    [P2,V2]=findRaysDarcy02(CalibFile,x_pxC2,y_pxC2,Ttype)
%%
tic
figure
histogram([structPotentialPairs.d])
toc
%% I give [x,y], it finds the closest track for both cameras
icam = 1;
x = 129;
y = 555;
clear d

if icam == 1
    for itrj = 1 : length(trajArray_CAM1)
        clear xt yt
        xt = trajArray_CAM1(itrj).track(:,1);
        yt = trajArray_CAM1(itrj).track(:,2);
        clear dptraj
        for ip = 1 : length(xt)
            dptraj(ip) = ( (x-xt(ip))^2 + (y-yt(ip))^2)^(1/2);
        end
        d(itrj) = min(dptraj);
    end
elseif icam == 2
    for itrj = 1 : length(trajArray_CAM2)
        clear xt yt
        xt = trajArray_CAM2(itrj).track(:,1);
        yt = trajArray_CAM2(itrj).track(:,2);
        clear dptraj
        for ip = 1 : length(xt)
            dptraj(ip) = ( (x-xt(ip))^2 + (y-yt(ip))^2)^(1/2);
        end
        d(itrj) = min(dptraj);
    end
    
end
[~,itrjClosest] = min(d);
itrjClosest
%% show the trajectories

figure('defaultAxesFontSize',20), hold on, box on

% for it = 1 : size(CC1,2)
%     clear xpos ypos
%     xpos = [CC1(it).X];
%     ypos = [CC1(it).Y];
%     plot(xpos,ypos,'ob')
%     
%     clear xpos ypos
%     xpos = part_cam2(it).pos(:,1);
%     ypos = part_cam2(it).pos(:,2);
%     plot(xpos,ypos,'or')
% end

for itrj = 1 : length(trajArray_CAM1)
    % camera 1
    clear xt yt
    xt = trajArray_CAM1(itrj).track(:,1);
    yt = trajArray_CAM1(itrj).track(:,2);
    plot(xt,yt,'-b','lineWidth',2)
    
end

for itrj = 1 : length(trajArray_CAM2)
    % camera 2
    clear xt yt
    xt = trajArray_CAM2(itrj).track(:,1);
    yt = trajArray_CAM2(itrj).track(:,2);
    plot(xt,yt,'-r','lineWidth',2)
end
%% testing the function
imageCorrelation(xm,ym,ACC1,ACC2,round(wti/2),filterOrder,'cleanC',dxPass01,dyPass01,R)
%% functions

function [P,V]=findRaysDarcy02(CalibFile,x_px,y_px,Ttype)
%% calib : calibration data for this camera
%% x_px  : x coordinates in px,
%% y_px  : y coordinates in px,
%% Ttype : type of the transformation to use (T1=Linear, T3=Cubic).

calibTemp = load(CalibFile,'calib'); calib = calibTemp.calib;

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

function [xyz0,direction]=fit3Dline_nan(XYZ)
%% [xyz0,direction]=fit3Dline_jv(XYZ)
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

% debug the cleaning of C (varargin in function)
function [xoffSet,yoffSet] = imageCorrelation(xc,yc,ACC1,ACC2,w,filterOrder,varargin)
% varargin: ,'cleanC',dxPass01,dyPass01,R);
ACC1sub = zeros(w+1,w+1,'uint8');
ACC1sub = ACC1(yc-w:yc+w,xc-w:xc+w);
C = normxcorr2(imgaussfilt(ACC1sub,filterOrder),imgaussfilt(ACC2,filterOrder));


% set C to zero above a predefined radius
% Checking varargin structure
if ( length(varargin) > 1 )
    fprintf('cleaning C \n')
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



%%