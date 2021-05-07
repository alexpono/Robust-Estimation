cd('C:\Users\darcy\Desktop\havingFun')
load('FIJI_test.mat')
%%
tmin = 0001;
tmax = 2000;
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
    idxt = find(FIJI_test(:,3)==it);
    hp = plot(FIJI_test(idxt,1),FIJI_test(idxt,2),'ok',...
        'MarkerEdgeColor','none','markerFaceColor',colP(it,:));
    %pause(.1)
end


%% find tracks and stitch them
tic
clear trajArray_CAM1 tracks_CAM1 part_cam1
for it = 1 : 2000
    idxt = find(FIJI_test(:,3)==it);
    part_cam1(it).pos(:,1) = [FIJI_test(idxt,1)]; % out_CAM1(:,1);
    part_cam1(it).pos(:,2) = [FIJI_test(idxt,2)]; % out_CAM1(:,2);
    part_cam1(it).pos(:,3) = ones(length([FIJI_test(idxt,1)]),1)*it;
    part_cam1(it).intensity = 0; %mI;
end

maxdist = 3;
longmin = 5;
[trajArray_CAM1,tracks_CAM1]=TAN_track2d(part_cam1,maxdist,longmin);
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
toc

%% take two trajectories and evaluate if they can be stitched
% i give a point it find the closest trajectory
tic
P1 = [983,570]; %[983,539];
P2 = [983,568]; %[985,543];
% loop on trajectories
clear d
d = zeros(1,length(trajArray_CAM1));
for itraj = 1 : length(trajArray_CAM1)
    clear xtrj ytrj 
    xtrj = trajArray_CAM1(itraj).track(:,1);
    ytrj = trajArray_CAM1(itraj).track(:,2);
    d1(itraj) = min(sqrt((P1(1)-xtrj).^2 + (P1(2)-ytrj).^2));
    d2(itraj) = min(sqrt((P2(1)-xtrj).^2 + (P2(2)-ytrj).^2));
end
[~,itrajPot1] = min(d1);
[~,itrajPot2] = min(d2);
toc

figure, hold on, box on
set(gca,'ydir','reverse')
xtrj1 = trajArray_CAM1(itrajPot1).track(:,1);
ytrj1 = trajArray_CAM1(itrajPot1).track(:,2);
xtrj2 = trajArray_CAM1(itrajPot2).track(:,1);
ytrj2 = trajArray_CAM1(itrajPot2).track(:,2);
plot(xtrj1,ytrj1)
plot(xtrj2,ytrj2)
%%
min(trajArray_CAM1(itrajPot1).track(:,3))
max(trajArray_CAM1(itrajPot1).track(:,3))
min(trajArray_CAM1(itrajPot2).track(:,3))
max(trajArray_CAM1(itrajPot2).track(:,3))
%%
clear Xtck Ytck tckSize
Xtck = []; Ytck = []; tckSize = [];
figure(ht), hold on
tic
for it = 1 : length(trajArray_CAM1)
    tckSize(it) = length(trajArray_CAM1(it).track(:,1));
end
toc

Xtck = NaN(length(trajArray_CAM1),max(tckSize));
Ytck = NaN(length(trajArray_CAM1),max(tckSize));

for it = 1 : length(trajArray_CAM1)
    Xtck(it,1:length(trajArray_CAM1(it).track(:,1))) = ...
        trajArray_CAM1(it).track(:,1);
    Ytck(it,1:length(trajArray_CAM1(it).track(:,1))) = ... 
        trajArray_CAM1(it).track(:,2);
end
htrck = plot(Xtck',Ytck','-','lineWidth',4);
%%



%%



%%



%%



%%



%%





