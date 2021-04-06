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

totalnFrames = 500;
him = 1152;
wim = 1152;
% load CC:
% cd('D:\IFPEN\IFPEN_manips\expe_2021_03_11\for4DPTV\re01_20spatules\Processed_DATA\zaber_100mm_20spatules_16bit_20210311T153131')
CCtemp = load('centers_cam1.mat', 'CC');
CC1 = CCtemp.CC;
CCtemp = load('centers_cam2.mat', 'CC');
CC2 = CCtemp.CC;
%% ROBUST ESTIMATION PART 1.1 removing the NaNs for all t
for it = 1 : 500 
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
hcam01 = figure;
imagesc(20*ACC1)
title('Camera1')
hcam02 = figure;
imagesc(20*ACC2)
title('Camera2')

%%  ROBUST ESTIMATION PART 1.3 normxcorr2 pass 01 (on a large window)
% xm,ym : fixed points in camera 1
c = clock; fprintf('start at %0.2dh%0.2dm\n',c(4),c(5)) 
filterOrder = 10;

% first pass
xm = round(wim/2);
ym = round(him/2);
wsub = 0.5*mean(xm,ym); % width correlation template image
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

%%  ROBUST ESTIMATION PART 1.3 normxcorr2 pass 02 (on a small window)

wti = 400; % width template images
wstep = 100; % step for sampling the image
nPartMin = 50; % minimum number of particles to calculate the correlation
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
                (1.5*dxPass01) + xc + wti/2 < wim && ...
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
        %patch('xdata',xp,'ydata',yp,'faceColor',pcol,'faceAlpha',.3,'edgeColor','none')
        %pause(.2)
        
        if tmpl_IM_tStr(iti).correlable == 1
    clear xm ym xoffSet yoffSet
    xm = tmpl_IM_tStr(iti).x;
    ym = tmpl_IM_tStr(iti).y;
    [xoffSet,yoffSet] = imageCorrelation(xm,ym,ACC1,ACC2,round(wti/2),filterOrder);%,'cleanC',dxPass01,dyPass01,R);
    tmpl_IM_tStr(iti).xoffSet = xoffSet;
    tmpl_IM_tStr(iti).yoffSet = yoffSet;
    quiver(xm,ym,xoffSet-xm,yoffSet-ym,'r','lineWidth',1)
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

[X,Y] = transformPointsForward(tform1,271,456);  % check some points
[X,Y] = transformPointsForward(tform1,0,0);           % check the change of (x0,y0)
ACC1tformed = imwarp(ACC2,tform1, 'OutputView', imref2d( size(ACC1) ));

falseColorOverlay = imfuse( 40*ACC1, 40*ACC1tformed);
imshow( falseColorOverlay, 'initialMagnification', 'fit');

% figure to check the tform1
% figure, hold on, box on
% for it = 1 : 500
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
for it = 1 : 500
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
%%
tic
maxdist = 3;
longmin = 10;
[trajArray_CAM1,tracks_CAM1]=TAN_track2d(part_cam1,maxdist,longmin);

maxdist = 6;
[trajArray_CAM2,tracks_CAM2]=TAN_track2d(part_cam2,maxdist,longmin);
toc

%% show the trajectories

figure('defaultAxesFontSize',20), hold on, box on

for it = 1 : 500
    clear xpos ypos
    xpos = [CC1(it).X];
    ypos = [CC1(it).Y];
    plot(xpos,ypos,'ob')
    
    clear xpos ypos
    xpos = part_cam2(it).pos(:,1);
    ypos = part_cam2(it).pos(:,2);
    plot(xpos,ypos,'or')
end

for itrj = 1 : length(trajArray_CAM1)
    % camera 1
    clear xt yt
    xt = trajArray_CAM1(itrj).track(:,1);
    yt = trajArray_CAM1(itrj).track(:,2);
    plot(xt,yt,'-b','lineWidth',2)
    
end

for itrj = 1 : length(trajArray_CAM1)
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
        %varargin = varargin{:,2}
        varargin{:,1}
        varargin{:,2}
        varargin{:,3}
        varargin{:,4}
        R = varargin{4};      % in pixels
        x0 = xc+varargin{2};
        y0 = yc+varargin{3};
        x = 1:1302;
        y = 1:1302;
        [xx,yy] = meshgrid(x,y);
        C(((xx-x0).^2+(yy-y0).^2) > R^2)=0;
    end

    %
    [ypeak,xpeak] = find(C==max(C(:)));
    yoffSet = ypeak-size(ACC1sub,1) + w;
    xoffSet = xpeak-size(ACC1sub,2) + w;
end