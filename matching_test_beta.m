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
% load CC:
CCtemp = load('centers_cam1.mat', 'CC');
CC1 = CCtemp.CC;
CCtemp = load('centers_cam2.mat', 'CC');
CC2 = CCtemp.CC;
%% ROBUST ESTIMATION PART 01 
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
%% ROBUST ESTIMATION PART 01 - normxcorr2 - we build the images

ACC1 = zeros(1152,1152,'uint8');
ACC2 = zeros(1152,1152,'uint8');
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
figure
imagesc(ACC1)
title('Camera1')
figure
imagesc(ACC2)
title('Camera2')
%% ROBUST ESTIMATION PART 01 -  select the squares for matching 
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
    xm = [xm,x];
    ym = [ym,y];
    drawrectangle(gca,'Position',[x-w,y-w,2*w,2*w], ...
            'FaceAlpha',0,'Color','b');
end


%% ROBUST ESTIMATION PART 01 -  normxcorr2 - we try to match CC1sub in CC2
%w = 100; % width correlating zone
filterOrder = 10;
transformationType = 'affine';

figure
imshow(20*ACC1)
hold on
for i = 1:length(xm)
drawrectangle(gca,'Position',[xm(i)-w,ym(i)-w,2*w,2*w], ...
    'FaceAlpha',0,'Color','b');
end
title('Selected rectangles from Cam1')


clear xc yc 
i=1
xc = xm(i)
yc = ym(i)
ACC1sub = zeros(w+1,w+1,'uint8');
ACC1sub = ACC1(yc-w:yc+w,xc-w:xc+w);
C = normxcorr2(imgaussfilt(ACC1sub,filterOrder),imgaussfilt(ACC2,filterOrder));
[ypeak,xpeak] = find(C==max(C(:)));
yoffSet = ypeak-size(ACC1sub,1);
xoffSet = xpeak-size(ACC1sub,2);
movingPoints=[yoffSet xoffSet];
fixedPoints = [yc xc]; %we take 3 points


figure
imshow(20*ACC2)
drawrectangle(gca,'Position',[xoffSet,yoffSet,size(ACC1sub,2),size(ACC1sub,1)], ...
'FaceAlpha',0,'Color','r');
title('Correlated rectangles from Cam1 in Cam2')
hold on
for i = 2:length(xm)
    clear xc yc 
    xc = xm(i);
    yc = ym(i);
    ACC1sub = zeros(w+1,w+1,'uint8');
    ACC1sub = ACC1(yc-w:yc+w,xc-w:xc+w);
    C = normxcorr2(imgaussfilt(ACC1sub,filterOrder),imgaussfilt(ACC2,filterOrder));
    [ypeak,xpeak] = find(C==max(C(:)));
    yoffSet = ypeak-size(ACC1sub,1);
    xoffSet = xpeak-size(ACC1sub,2);
    movingPoints=cat(1,movingPoints,[yoffSet xoffSet]);
    fixedPoints = cat(1,fixedPoints,[yc xc]); %we take 3 points
    drawrectangle(gca,'Position',[xoffSet,yoffSet,size(ACC1sub,2),size(ACC1sub,1)], ...
    'FaceAlpha',0,'Color','r');
end

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