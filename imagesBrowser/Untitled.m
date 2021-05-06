
cd('D:\IFPEN\IFPEN_manips\expe_2021_04_20_beads\for4DPTV\DATA\expe65_20210420T172713\cam1')
listTif = dir('*.tif');


it = 1000;
low_in = 2/255;
high_in = 15/255;
th = 3;
sz = 3;

A = imread(listTif(it).name);



out=pkfnd(A,th,sz);

figure('defaultAxesFontSize',20)
imshow(imadjust(A,[low_in , high_in]))
hold on
plot(out(:,1),out(:,2),'or')

%% navigate in time
close all


him = figure('defaultAxesFontSize',20); hold on, box on

it = 1000;
low_in = 2/255;
high_in = 15/255;
th = 3;
sz = 3;

[widthIm,heightIm] = size(imread(listTif(it).name));
himt = imshow( imadjust( imread(listTif(it).name),[low_in , high_in]) );
xAxis = [0,widthIm];
yAxis = [0,heightIm];

zmfctr = widthIm; % zoom factor

clear out
out=pkfnd(A,th,sz);

figure(him), hold on
hprt = plot(out(:,1),out(:,2),'ob');
set(gcf,'currentchar','a')         % set a dummy character
set(gcf,'position',[100 100 900 800])
title(sprintf('time: %0.0f',it))

% hminiMap = figure('defaultAxesFontSize',20); hold on, box on
% plot3([0,336,336,0,0,0,336,336,0,0],[0,0,336,336,0,0,0,336,336,0],2016*[0,0,0,0,0,1,1,1,1,1],'.k')
% X = [0,336,336,0,0];
% Y = [0,0,336,336,0];
% Z = iz * [1,1,1,1,1];
% hp = patch('XData',X,'YData',Y,'ZData',Z);
% hp.FaceColor = 'r';
% hp.EdgeColor = 'none';
% hp.FaceAlpha = .5;
% set(gcf,'position',[1100 100 200 800])
% caz =  -36.4150;
% cel =    7.2736;
% view(caz,cel)
%    hold on
%    for ich = 1 : size(channels,2)
%        plot3([channels(ich).x],[channels(ich).y],[channels(ich).z],'o')
%    end

while contains('azesdxcrt',get(gcf,'currentchar'))  % which gets changed when key is pressed
    figure(him)
    [x,y] = ginput(1);
    if get(gcf,'currentchar')=='z'
        it = it -  1;
    elseif get(gcf,'currentchar')=='e'
        it = it +  1;
    elseif get(gcf,'currentchar')=='s'
        it = it - 10;
    elseif get(gcf,'currentchar')=='d'
        it = it + 10;
    elseif get(gcf,'currentchar')=='x'
        it = it - 100;
    elseif get(gcf,'currentchar')=='c'
        it = it + 100;
    elseif get(gcf,'currentchar')=='r'
        zmfctr = round(zmfctr/5);
        xAxis(1) = x-zmfctr; xAxis(2)=x+zmfctr; 
        yAxis(1) = y-zmfctr; yAxis(2)=y+zmfctr;
    elseif get(gcf,'currentchar')=='t'
        zmfctr = round(zmfctr*5);
        xAxis(1) = x-zmfctr; xAxis(2)=x+zmfctr; 
        yAxis(1) = y-zmfctr; yAxis(2)=y+zmfctr;
    end
    
    imt = imread(listTif(it).name);
    clear out
    out = pkfnd(imt,th,sz);
    
    figure(him), hold on
    himt.CData = imadjust(imt,[low_in , high_in]);
    axis(gca,[xAxis(1) xAxis(2) yAxis(1) yAxis(2)])
    hprt.XData = out(:,1);
    hprt.YData = out(:,2);
    %set(gcf,'position',[100 100 1000 1000])
    title(sprintf('time: %0.0f',it))
    
    %    figure(hminiMap)
    %    hp.Vertices(:,3) = iz * [1,1,1,1,1];
    
end

