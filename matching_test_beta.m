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


totalnFrames = 500;
him = 1152;
wim = 1152;
% load CC:
% cd('D:\IFPEN\IFPEN_manips\expe_2021_03_11\for4DPTV\re01_20spatules\Processed_DATA\zaber_100mm_20spatules_16bit_20210311T153131')
CCtemp = load('centers_cam1.mat', 'CC');
CC1 = CCtemp.CC;
CCtemp = load('centers_cam2.mat', 'CC');
CC2 = CCtemp.CC;
%% ROBUST ESTIMATION PART 01 - 01
%removing the NaNs for all t
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
%% ROBUST ESTIMATION PART 1.2 - normxcorr2 - we build the images

ACC1 = zeros(him,wim,'uint8');
ACC2 = zeros(him,wim,'uint8');
for it = 1 : totalnFrames
    for ip = 1 : length(CC1(it).X)
        xim1 = round(CC1(it).X(ip));
        yim1 = round(CC1(it).Y(ip));
        ACC1(yim1,xim1) = ACC1(yim1,xim1) + 100;
    end
        for ip = 1 : length(CC2(it).X)
        xim2 = round(CC2(it).X(ip));
        yim2 = round(CC2(it).Y(ip));
        ACC2(yim2,xim2) = ACC1(yim2,xim2) + 100;
    end
end
hcam01 = figure;
imagesc(ACC1)
title('Camera1')
hcam02 = figure;
imagesc(ACC2)
title('Camera2')

%%  ROBUST ESTIMATION PART 1.3 normxcorr2 smaller and smaller windows
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
dxPass01 =   xoffSet-xm
dyPass01 =   yoffSet-ym
R = (dxPass01^2+dyPass01^2)^(1/2) 
c = clock; fprintf('finished at %0.2dh%0.2dm\n',c(4),c(5)) 

% second pass

%% ROBUST ESTIMATION PART 01 -  normxcorr2 - we try to match CC1sub in CC2
%w = 100; % width correlating zone
c = clock; fprintf('start at %0.2dh%0.2dm\n',c(4),c(5)) 

transformationType = 'affine';

clear xMP yMP % moving points
xMP = []; yMP = [];

figure
imshow(20*ACC1)
hold on
for i = 1:length(xm)
    xc = xm(i);
    yc = ym(i);
    drawrectangle(gca,'Position',[xm(i)-w,ym(i)-w,2*w,2*w], ...
        'FaceAlpha',0,'Color','b');
end
title('Selected rectangles from Cam1')

clear xc yc 
hcam02 = figure;
imshow(20*ACC2)
hold on, box on
%     drawrectangle(gca,'Position',[xoffSet,yoffSet,size(ACC1sub,2),size(ACC1sub,1)], ...
%         'FaceAlpha',0,'Color','r');
%     title('Correlated rectangles from Cam1 in Cam2')
%     hold on

for i = 1 : length(xm)
    
    drawrectangle(gca,'Position',[xoffSet,yoffSet,size(ACC1sub,2),size(ACC1sub,1)], ...
        'FaceAlpha',0,'Color','r');
    xMP = [xMP,xoffSet+w];
    yMP = [yMP,yoffSet+w];
end
c = clock; fprintf('done at %0.2dh%0.2dm\n',c(4),c(5)) 
%%
clear d dx dy
for ip = 1 : length(xMP)
    d(ip)  = ((xm(ip)-xMP(ip))^2 + (ym(ip)-yMP(ip))^2)^(1/2);
    dx(ip) = -xm(ip)+xMP(ip);
    dy(ip) = -ym(ip)+yMP(ip);
end
figure('defaultAxesFontSize',20);
histogram(d,[0:50:1000])
title('d')
figure('defaultAxesFontSize',20);
histogram(dx)%,[0:50:1000])
title('dx')
figure('defaultAxesFontSize',20);
histogram(dy)%,[0:50:1000])
title('dy')
%% plot hitogram of x and y shift
hhist = figure('defaultAxesFontSize',20);
set(gca,'ydir','reverse')
hold on, box on
for i = 1 : length(xm)
    plot([xm(i)],[ym(i)],'o')
    plot([xm(i),xMP(i)],[ym(i),yMP(i)])
end
axis([1 1152 1 1152])
hquiver = figure('defaultAxesFontSize',20);
imshow(20*ACC1)
set(gca,'ydir','reverse')
hold on, box on
quiver(xm,ym,xMP-xm,yMP-ym)
axis([1 1152 1 1152])
%%

tform = fitgeotrans(double(fixedPoints),double(movingPoints),transformationType);
ACC2_transformed = imwarp(ACC2,tform,'OutputView',imref2d(size(ACC1)));
%%
figure
imshowpair(ACC1,ACC2_transformed)
%imshow(ACC2_transformed)

%% test
fixedPoints = [100 200 400 100 ; 200 400 300 400]';
movingPoints = [200 300 500 200 ; 200 400 300 400]';

tform = fitgeotrans(double(movingPoints),double(fixedPoints),transformationType);
ACC2_transformed = imwarp(ACC2,tform,'OutputView',imref2d(size(ACC1)));

falseColorOverlay = imfuse( 40*ACC1, 40*ACC1tformed);
imshow( falseColorOverlay, 'initialMagnification', 'fit');


%%

figure
imshow(20*ACC2)
hold on
for i = 1:length(xm)
drawrectangle(gca,'Position',[xm(i)-w,ym(i)-w,2*w,2*w], ...
    'FaceAlpha',0,'Color','b');
end
title('Selected rectangles from Cam1')

%%
fixedPoints = [163 427 963 951; 570 781 322 738]';
movingPoints = [271 522 1002 1002; 456 631 198 573]';

transformationType = 'affine';
tform = fitgeotrans(movingPoints,fixedPoints,transformationType);

[X,Y] = transformPointsForward(tform,271,456);  % check some points
[X,Y] = transformPointsForward(tform,0,0);           % check the change of (x0,y0)
ACC1tformed = imwarp(ACC2,tform, 'OutputView', imref2d( size(ACC1) ));

falseColorOverlay = imfuse( 40*ACC1, 40*ACC1tformed);
imshow( falseColorOverlay, 'initialMagnification', 'fit');

<<<<<<< HEAD
%%
figure, hold on, box on
for it = 1 : 500
    %pause(.1)
    inputPoints = [CC2(it).X;CC2(it).Y]';
    PointsC1 = [CC1(it).X;CC1(it).Y]';
    [X,Y] = transformPointsForward(tform,inputPoints(:,1),inputPoints(:,2));
    
    plot(X,Y,'or')
    plot(PointsC1(:,1),PointsC1(:,2),'ob')
    pause(.1)
end
=======
%% OLD Stuff %%


%% ROBUST ESTIMATION PART 1.3 - OLD - ginput select the squares for matching 
w = 75;
figure
imshow(imgaussfilt(255-20*ACC1,1))

% matching coordinates
xm = [];
ym = [];
while(1)
    clear x y
    [x,y] = ginput(1);
    if x<0
        break
    end
    xm = [xm,round(x)];
    ym = [ym,round(y)];
    drawrectangle(gca,'Position',[x-w,y-w,2*w,2*w], ...
            'FaceAlpha',0,'Color','b');
end


>>>>>>> 28e2165535b8b1782ee8191c1eb079153ba4029a
%%
% %% a code that could replace ginput, it finds potential correlation zones by itself
% close all 
% clear x4corr y4corr
% h0 = figure;
% imagesc(50*ACC1), colormap gray
% 
% smoothFactor = 8;
% h1 = figure;
% ACC1gaussed = imgaussfilt(ACC1,smoothFactor);
% imagesc(ACC1gaussed)
% Localmax = imregionalmax(ACC1gaussed,8);
% h2 = figure;
% imagesc(Localmax)
% stats = regionprops(Localmax,'centroid','Area');
% % h3 = figure;
% % histogram([stats.Area],[0:50:5000])
% 
% hold on
% icz = 0; % cz for correlation zone
% for is = 1 : size(stats,1)
%     if stats(is).Area > 250 || ACC1gaussed(round(stats(is).Centroid(2)),round(stats(is).Centroid(1))) > 1
%         icz = icz + 1;
%         x4corr(icz) = stats(is).Centroid(1);
%         y4corr(icz) = stats(is).Centroid(2);
%     end
% end
% 
% figure(h0), hold on
% plot(x4corr,y4corr,'s')
% figure(h1), hold on
% plot(x4corr,y4corr,'sr')
% figure(h2), hold on
% plot(x4corr,y4corr,'s')

%% functions

function [xoffSet,yoffSet] = imageCorrelation(xc,yc,ACC1,ACC2,w,filterOrder)
    ACC1sub = zeros(w+1,w+1,'uint8');
    ACC1sub = ACC1(yc-w:yc+w,xc-w:xc+w);
    C = normxcorr2(imgaussfilt(ACC1sub,filterOrder),imgaussfilt(ACC2,filterOrder));
    

    % set C to zero above a predefined distance
%     R = 300;
%     x0 = xc+w;
%     y0 = yc+w;
%     x = 1:1302;
%     y = 1:1302;
%     [xx,yy] = meshgrid(x,y);
%     C(((xx-x0).^2+(yy-y0).^2) > R^2)=0;
    %
    [ypeak,xpeak] = find(C==max(C(:)));
    yoffSet = ypeak-size(ACC1sub,1) + w;
    xoffSet = xpeak-size(ACC1sub,2) + w;
end