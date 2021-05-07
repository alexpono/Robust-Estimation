%% WORKFLOW ..

% detect the computer and load all_IFPEN_DARCY02_experiments
name = getenv('COMPUTERNAME');
if strcmp(name,'DESKTOP-3ONLTD9')
    cd('C:\Users\Lenovo\Jottacloud\RECHERCHE\Projets\21_IFPEN\git\Robust-Estimation')
elseif strcmp(name,'DARCY')
    cd('C:\Users\darcy\Desktop\git\Robust-Estimation')
end
load('all_IFPEN_DARCY02_experiments.mat')


%% STEP 4 - calib manip 2021 04 22
dirIn  = 'D:\IFPEN\IFPEN_manips\expe_2021_05_06_calibration\images4calibration\';
zPlanes = [00:05:40]; % mm
camName = {1,2};
dotSize = 30; % pixels
th = 60; 

extension = 'tif';
FirstPlane = 1;
FirstCam   =  1;

gridSpace = 5;            % mm
lnoise = 1;
blackDots = 0;
%MakeCalibration_Vnotch(dirIn,zPlanes,camName,gridSpace,th,dotSize,lnoise,blackDots,extension,FirstPlane,FirstCam)
% manip IFPEN
cd(dirIn)
MakeCalibration_Vnotch(dirIn, zPlanes, camName, gridSpace, th, dotSize, lnoise, blackDots,extension,FirstPlane,FirstCam)

%% STEP 0 - Defining paths using list of experiments stored in sturture allExpeStrct

iexpe = 2;

allExpeStrct(iexpe).type        = 'experiment';
allExpeStrct(iexpe).name        = 'expe20210505_run03';
allExpeStrct(iexpe).inputFolder = ...
    strcat('D:\IFPEN\IFPEN_manips\expe_2021_05_05\run03\');
allExpeStrct(iexpe).analysisFolder = ...
    strcat('D:\pono\IFPEN\analysisExperiments\analysis_expe_20210505\'); 
allExpeStrct(iexpe).CalibFile = ...
    strcat('D:\IFPEN\IFPEN_manips\expe_2021_05_06_calibration\',...
           'images4calibration\calib.mat');
       
allExpeStrct(iexpe).centerFinding_th = 2; % automatiser la définition de ces paramètres?
allExpeStrct(iexpe).centerFinding_sz = 2; % automatiser la définition de ces paramètres?

%%
fprintf('define folders and experiment name\n')
nameExpe = 'expe65_20210420T172713';
inputFolder = 'D:\IFPEN\IFPEN_manips\expe_2021_04_20_beads\';
outputFolder = strcat(inputFolder,'for4DPTV\DATA\',nameExpe);

session.input_path = strcat(inputFolder,'for4DPTV\');
session.output_path = session.input_path;

analysisFolder = 'D:\pono\IFPEN\analysisExperiments\analysis_expe_20210311\analysis4DPTV';
sessionPath = analysisFolder;                       % for 4D PTV

% CalibFile = strcat('D:\pono\IFPEN\IFPEN_manips\expe_2021_02_16_calibration\',...
%            'calibration_mcin2\images4_4DPTV\calib.mat');
CalibFile = strcat('D:\IFPEN\IFPEN_manips\expe_2021_04_22_calibration\for4DPTV\calib.mat');       
    
fprintf('define folders and experiment name - DONE \n')

%% findTracks

maxdist = 3;
longmin = 5;
iexpe = 2;
for iSeq = 35:35 % loop on images sequences

[trajArray_CAM1,tracks_CAM1] = ...,
    DARCY02_findTracks(allExpeStrct,iexpe,iSeq,maxdist,longmin);

end

%% temp
% clear xA yA xB yB
% figure(hFIJI)
% figure(ht)
% figure(hFIJI)
% [xA,yA] = ginput(1);
% plot(xA,yA,'og')
% [xB,yB] = ginput(1);
% xlim([xA xB]) 
% ylim([yA yB])
% figure(ht)
% xlim([xA xB]) 
% ylim([yA yB])
% 
% clear xA yA xB yB











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
%%
min(min(XYZ(:,1,:)))
max(max(XYZ(:,1,:)))

min(min(XYZ(:,2,:)))
max(max(XYZ(:,2,:)))

min(min(XYZ(:,3,:)))
max(max(XYZ(:,3,:)))
%%
figure('defaultAxesFontSize',20)
histogram([XYZ(:,1,:)])
xlabel('X')
figure('defaultAxesFontSize',20)
histogram([XYZ(:,2,:)])
xlabel('Y')
figure('defaultAxesFontSize',20)
histogram([XYZ(:,3,:)])
xlabel('Z')
%% showing PV
figure('defaultaxesFontSize',20)
hold on
box on
lray = 4000;
for icam = 1 : 2
    if icam == 1
        col = 'b';
    elseif icam ==2
        col = 'r';
    end
    clear P V
    P = (camPV(icam,it).P);
    V = (camPV(icam,it).V);
    for ip = 1 : size(P,1)
        ip
        clear X Y Z XR YR ZR
        X = P(ip,1);
        Y = P(ip,2);
        Z = P(ip,3);
        XR = P(ip,1)+[-lray*V(ip,1),lray*V(ip,1)];
        YR = P(ip,2)+[-lray*V(ip,2),lray*V(ip,2)];
        ZR = P(ip,3)+[-lray*V(ip,3),lray*V(ip,3)];
        plot3(X,Y,Z,'o','color',col)
        plot3(XR,YR,ZR,'--b','color',col)
    end
end
view(3)
xlabel('x')
ylabel('y')
zlabel('z')
axis([-1000 1000 -1000 1000 -1000 2000])
plot3(X3D,Y3D,Z3D,'og')
axis([-200 200 -200 200 -60 60])
axis equal
%% STEP 7 - BLP
tic

dmax12 = 5%0.5; % cquoiça !!!

clear X3D Y3D Z3D d3D
X3D = []; Y3D = []; Z3D = []; d3D = [];

icam = 1;
Pcam1 = camPV(icam,it).P;
Vcam1 = camPV(icam,it).V;
icam = 2;
Pcam2 = camPV(icam,it).P;
Vcam2 = camPV(icam,it).V;

p = Pcam2;
v = Vcam2;
for ipcam1 = 1 : size(Pcam1,1)
    p1 = Pcam1(ipcam1,:);
    v1 = Vcam1(ipcam1,:);
    [cent,dist] = TAN_closest_point_to_lines2(p1,v1,p,v);
    [a,b] = min(dist);
    if a < dmax12
        X3D = [X3D,cent(b,1)];
        Y3D = [Y3D,cent(b,2)];
        Z3D = [Z3D,cent(b,3)];
        d3D = [d3D,a];
    end
end
toc

figure('defaultaxesFontSize',20)
hold on
box on
plot3(X3D,Y3D,Z3D,'og')
view(3)
xlabel('x')
ylabel('y')
zlabel('z')
%%
figure('defaultaxesFontSize',20)
box on
histogram(Z3D,[10:.1:60])

%% STEP 6 - center to rays
tic
camID = [1,2];
[P, V] = Centers2Rays(session,nameExpe,CalibFile,camID);
toc

%% step 6 - trying to show the rays

cd(session.output_path)
cd('Processed_DATA')
cd(nameExpe)

load('rays.mat')

%% showing rays - to remove . . .
figure
hold on, box on

it   = 1;

iCam = 1;
for ip = 1 : length(datacam(iCam).data(it).P(:,1))
X = datacam(iCam).data(it).P(ip,1)*[1-10*datacam(iCam).data(it).V(ip,1),1+10*datacam(iCam).data(it).V(ip,1)];
Y = datacam(iCam).data(it).P(ip,2)*[1-10*datacam(iCam).data(it).V(ip,2),1+10*datacam(iCam).data(it).V(ip,2)];
Z = datacam(iCam).data(it).P(ip,3)*[1-10*datacam(iCam).data(it).V(ip,3),1+10*datacam(iCam).data(it).V(ip,3)];
plot3(X(1),Y(1),Z(1),'ob')
plot3(X,Y,Z,'-b')
end
iCam = 2;
for ip = 1 : length(datacam(iCam).data(it).P(:,1))
X = datacam(iCam).data(it).P(ip,1)*[1,1+10*datacam(iCam).data(it).V(ip,1)];
Y = datacam(iCam).data(it).P(ip,2)*[1,1+10*datacam(iCam).data(it).V(ip,2)];
Z = datacam(iCam).data(it).P(ip,3)*[1,1+10*datacam(iCam).data(it).V(ip,3)];
plot3(X(1),Y(1),Z(1),'or')
plot3(X,Y,Z,'-r')
end
view(3)
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

%%
figure, hold on, box on
for it = 1 : length(traj)
    clear x3 y3 z3
    x3 = traj(it).x;
    y3 = traj(it).y;
    z3 = traj(it).z;
    plot3(x3,y3,z3)
end
xlabel('x')
ylabel('y')
zlabel('z')
%% step 9 - Stitching

dfmax = 4; % maximum number of tolerated missing frames to reconnect to trajectories
dxmax = 2*0.031; % (mm) % maximum tolerated distance between stitchs parts
dvmax = 0.3;
lmin  = 2*0.031;
StitchedTraj = Stitching(session,nameExpe,trackName,dfmax,dxmax,dvmax,lmin);




%% FUNCTIONS 

%%

%%

%%

function [trajArray_CAM1,tracks_CAM1] = DARCY02_findTracks(allExpeStrct,iexpe,ifile,maxdist,longmin)

fprintf('load image sequence \n')

inputFolder = allExpeStrct(iexpe).inputFolder;
cd(inputFolder)

cd(inputFolder)
listMcin2 = dir('*.mcin2');
filename  = listMcin2(ifile).name;

cd(inputFolder)
[~,~,params] = mCINREAD2(filename,1,1);

% save images as .tif image sequence
totalnFrames = params.total_nframes;
cd(inputFolder)
[M,~,params]=mCINREAD2(filename,1,totalnFrames);

fprintf('load image sequence - DONE \n')

% calculate mean image
tic
ImMean = uint8(mean(M,3));
%ImMax = size(ImMean);
ImMax  = max(M,[],3);
toc
figure
imagesc(ImMean)
figure
imagesc(ImMax)
%
Im01 = M - ImMean;
imhist(Im01)

%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%
% determine particules positions
th = allExpeStrct(iexpe).centerFinding_th;
sz = allExpeStrct(iexpe).centerFinding_sz;
tic
for it = 1 : size(M,3)
    CC(it).xy = pkfnd(Im01(:,:,it),th,sz);
end
toc
figure
imagesc(Im01(:,:,1))
hold on
plot(CC(1).xy(:,1),CC(1).xy(:,2),'ob')

%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%
clear CCall %= [];
for it = 1 : size(M,3)
    X = CC(it).xy(:,1);
    Y = CC(it).xy(:,2);
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

%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%
% find tracks and stitch them

tic
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
toc

%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%

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

end


