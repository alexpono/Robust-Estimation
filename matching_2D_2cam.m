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

him = 1152;
wim = 1152;
% load CC:
% cd('D:\IFPEN\IFPEN_manips\expe_2021_03_11\for4DPTV\re01_20spatules\Processed_DATA\zaber_100mm_20spatules_16bit_20210311T153131')

%cd('C:\Users\Lenovo\Desktop\IFPEN\DL\for4DPTV\Processed_DATA\visu01_20210402T160947')
cd('C:\Users\Lenovo\Desktop\IFPEN\DL\for4DPTV\Processed_DATA\visu01_20210402T155814')
CCtemp = load('centers_cam1.mat', 'CC');
CC1 = CCtemp.CC;
CCtemp = load('centers_cam2.mat', 'CC');
CC2 = CCtemp.CC;
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
nPartMin = 100; % minimum number of particles to calculate the correlation
tmpl_IM_tStr = struct(); % structure storing information on template images

% cut the image in a lot of small images
hcam01 = figure;
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
        patch('xdata',xp,'ydata',yp,'faceColor',pcol,'faceAlpha',.3,'edgeColor','none')
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
                quiver(xm,ym,xoffSet-xm,yoffSet-ym,'--r','lineWidth',2)
            end
        end
    end
end


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
[X,Y] = transformPointsForward(tform1,0,0);           % check the change of (x0,y0)
ACC1tformed = imwarp(ACC2,tform1, 'OutputView', imref2d( size(ACC1) ));

falseColorOverlay = imfuse( 40*ACC1, 40*ACC1tformed);
imshow( falseColorOverlay, 'initialMagnification', 'fit');

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

maxdist = 6;
[trajArray_CAM2,tracks_CAM2]=TAN_track2d(part_cam2,maxdist,longmin);
toc

%% try to associate trajectories: % -> SPEED UP THIS PART !!!
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
%%
c = clock; fprintf('start at %0.2dh%0.2dm\n',c(4),c(5))
tic
figure('defaultAxesFontSize',20), box on, hold on
%%%%%%%%%%
for iP = 1 : length(structPotentialPairs)
    iP
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
        plot([x01(it1),x02(it2)],[y01(it1),y02(it2)],'-g')
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
%%
figure, hold on
 p1 = plot(rand(10,1),'r-','LineWidth',5); hold on
 p2 = plot(rand(10,1),'r-','LineWidth',2);
 p1.Color(4) = 0.25;
 p2.Color(4) = 0.75;
%% good and bad pairing
tic
c = clock; fprintf('start at %0.2dh%0.2dm\n',c(4),c(5))
figure('defaultAxesFontSize',10), box on, hold on
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
if ddd < 10 && (abs(Dx01-Dx02)<5) && (abs(Dy01-Dy02)<5)
    plot(x01,y01,'-b','lineWidth',2)
    plot(x02,y02,'-r','lineWidth',2)
    for it = 1 : length([structPotentialPairs(iP).tCAM01])
        it1 = structPotentialPairs(iP).tCAM01(it);
        it2 = structPotentialPairs(iP).tCAM02(it);
        plot([x01(it1),x02(it2)],[y01(it1),y02(it2)],'-g')
    end
end
%title(sprintf('iP: %0.0f, d: %0.0f, Dx01: %0.0f, Dx02: %0.0f, Dy01: %0.0f, Dy02: %0.0f ',...
%    iP,structPotentialPairs(iP).d,Dx01, Dx02, Dy01, Dy02))
end
toc
c = clock; fprintf('done at %0.2dh%0.2dm\n',c(4),c(5))
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