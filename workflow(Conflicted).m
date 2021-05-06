%% WORKFLOW ..

cd('C:\Users\darcy\Desktop\git\Robust-Estimation')
load('all_IFPEN_DARCY02_experiments.mat')

%% STEP 0 - Defining paths using list of experiments stored in sturture allExpeStrct

% iexpe = 1;
% allExpeStrct(iexpe).name        = 'visu01_20210402T155814';
% allExpeStrct(iexpe).inputFolder = ...
%     strcat('D:\IFPEN\IFPEN_manips\expe_2021_04_02_withouBeads\');
% allExpeStrct(iexpe).analysisFolder = ...
%     strcat('D:\pono\IFPEN\analysisExperiments\analysis_expe_20210402\'); 
% allExpeStrct(iexpe).CalibFile = ...
%     strcat('D:\IFPEN\IFPEN_manips\expe_2021_02_16_calibration\',...
%            'calibration_mcin2\images4_4DPTV\calib.mat');
% allExpeStrct(iexpe).centerFinding_th = 1;
% allExpeStrct(iexpe).centerFinding_sz = 1;

allExpeStrct(iexpe).CalibFile
fprintf('define folders and experiment name\n')
nameExpe = allExpeStrct(iexpe).name;
inputFolder = allExpeStrct(iexpe).inputFolder;
outputFolder = strcat(inputFolder,'for4DPTV\DATA\',nameExpe);
session.input_path = strcat(inputFolder,'for4DPTV\');
session.output_path = session.input_path;
analysisFolder = allExpeStrct(iexpe).analysisFolder;
sessionPath = analysisFolder;                       % for 4D PTV
CalibFile = allExpeStrct(iexpe).CalibFile;
fprintf('defined folders and experiment name\n')

%% STEP 0 - Defining paths 

tic
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
       
toc


%% STEP 0 - Defining paths _ experiment expe_20210217_re3_zaber825

tic
fprintf('define folders and experiment name\n')



nameExpe = 'expe_20210217_re3_zaber825';
inputFolder = 'D:\IFPEN\IFPEN_manips\expe_2021_02_17\';

outputFolder = strcat(inputFolder,'for4DPTV\DATA\',nameExpe);

session.input_path = strcat(inputFolder,'for4DPTV\');
session.output_path = session.input_path;

analysisFolder = 'D:\IFPEN\analysisExperiments\expe_20210217\';
sessionPath = analysisFolder;                       % for 4D PTV

% CalibFile = strcat('D:\pono\IFPEN\IFPEN_manips\expe_2021_02_16_calibration\',...
%            'calibration_mcin2\images4_4DPTV\calib.mat');
CalibFile = strcat('D:\IFPEN\IFPEN_manips\expe_2021_02_16_calibration\calibration_mcin2\images4_4DPTV\calib.mat');       
       
toc

%% STEP 0 - Defining paths
tic
fprintf('define folders and experiment name\n')

nameExpe = 'visu01_20210402T155814';
inputFolder = 'C:\Users\Lenovo\Desktop\IFPEN\DL';
outputFolder = strcat('C:\Users\Lenovo\Desktop\IFPEN\DL\for4DPTV\DATA\',nameExpe);

session.input_path = strcat('C:\Users\Lenovo\Desktop\IFPEN\DL\for4DPTV\');
session.output_path = session.input_path;

analysisFolder = 'C:\Users\Lenovo\Desktop\IFPEN\DL\analysisFolder';
sessionPath = analysisFolder;                       % for 4D PTV

CalibFile = strcat('expe_20210216_calibration\calib.mat');
toc
%%  STEP 0 - Defining paths - papier déplacé à la main
tic
fprintf('define folders and experiment name\n')

nameExpe = 'expe_20210323_alamain01';
dataFolder = 'D:\pono\IFPEN\IFPEN_manips\expe_2021_03_23_alamain\DATA\';
inputFolder = 'D:\pono\IFPEN\IFPEN_manips\expe_2021_03_23_alamain';
outputFolder = strcat(dataFolder,nameExpe);
cd(dataFolder), mkdir(nameExpe)

session.input_path = strcat(outputFolder);
session.output_path = session.input_path;

analysisFolder = 'D:\pono\IFPEN\analysisExperiments\analysis_expe_20210323';
sessionPath = analysisFolder;                       % for 4D PTV

CalibFile = strcat('D:\pono\IFPEN\IFPEN_manips\expe_2021_02_16_calibration\',...
           'calibration_mcin2\images4_4DPTV\calib.mat');
toc

%% STEP 0 - from .mcin2 to .tif image sequences for 4DPTV analysis

cd(inputFolder)
clear files
files = struct();
namesfiles = dir('*.mcin2');
%%
files(1).name = namesfiles(1).name;
files(2).name = namesfiles(2).name;


cd(outputFolder)
mkdir('cam1'), mkdir('cam2')
% files = dir('*.mcin2');
for iCam = 1 : 2 %length(files)
    filename = files(iCam).name;
    
    cd(inputFolder)
    [M,~,params]=mCINREAD2(filename,1,1);
        
    % save images as .tif image sequence
    totalnFrames = params.total_nframes;
    cd(inputFolder)
    [M,~,params]=mCINREAD2(filename,1,totalnFrames);
    
    for k = 1:totalnFrames 
        if (~mod(k,100) == 1) || (k==1)
            fprintf('image: %0.0f / %0.0f \n',k-200,totalnFrames)
        end
        cd(strcat(outputFolder,strcat('\',sprintf('cam%0.1d',iCam))))
        imwrite(M(:,:,k),strcat(nameExpe,sprintf('_cam%0.1d_%0.5d',iCam,k),'.tif'));
    end
    
end

fprintf('done\n')

%% STEP 4 - calib manip 2021 02 17
% dirIn  = 'D:\pono\IFPEN\IFPEN_manips\expe_2021_02_16_calibration\calibration_mcin2\images4_4DPTV\';
dirIn  = 'D:\pono\IFPEN\IFPEN_manips\expe_2021_02_16_calibration\calibration_mcin2\images4_4DPTV\';
zPlanes = [10,20,30,40,50]; % mm
camName = {1,2};
dotSize = 15; % pixels
th = 20; 

gridSpace = 2*5;            % mm
lnoise = 1;
blackDots = 0;
%MakeCalibration_Vnotch(dirIn,zPlanes,camName,gridSpace,th,dotSize,lnoise,blackDots,extension,FirstPlane,FirstCam)
% manip IFPEN
cd(dirIn)
MakeCalibration_Vnotch(dirIn, zPlanes, camName, gridSpace, th, dotSize, lnoise, blackDots)

%% STEP 4 - calib manip 2021 04 22
dirIn  = 'D:\IFPEN\IFPEN_manips\expe_2021_04_22_calibration\for4DPTV\';
zPlanes = [10:2:28]; % mm
camName = {1,2};
dotSize = 40; % pixels
th = 15; 

extension = 'tif';
FirstPlane = 10;
FirstCam   =  1;

gridSpace = 5;            % mm
lnoise = 1;
blackDots = 0;
%MakeCalibration_Vnotch(dirIn,zPlanes,camName,gridSpace,th,dotSize,lnoise,blackDots,extension,FirstPlane,FirstCam)
% manip IFPEN
cd(dirIn)
MakeCalibration_Vnotch(dirIn, zPlanes, camName, gridSpace, th, dotSize, lnoise, blackDots,extension,FirstPlane,FirstCam)

%% STEP 5 - background images - part 01 - background images

fprintf('calculate background images\n')

tic
BackgroundComputation(session,nameExpe,1,1,totalnFrames) % for camera 1
BackgroundComputation(session,nameExpe,2,1,totalnFrames) % for camera 2
toc
fprintf('done \n')

%% STEP 5 - background images - part 02 - showing the background images
cd(session.output_path)
cd('Processed_DATA')
cd(nameExpe)

load('Background_cam2.mat')

figure
imagesc(BackgroundMin)%, colormap gray
title('BackgroundMin')

figure
imagesc(BackgroundMax)%, colormap gray
title('BackgroundMax')

figure
imagesc(BackgroundMean)%, colormap gray
title('BackgroundMean')

%% STEP 5 - CenterFinding2D   - test mode

% session
CamNum      = 1;
firstFrame  = 1;
nframes     = totalnFrames;
th          = 1;
sz          = 1; 
test        = 1;
%BackgroundType
%format

CC = CenterFinding2D(session,nameExpe,CamNum,firstFrame,nframes,th,sz,test);
%% testing centerfinding2D
ManipName = nameExpe;
format='%05d.tif';
kframe = 1;
fprintf(ManipName);
folderin = fullfile(session.input_path, 'DATA', ManipName, ['cam' num2str(CamNum)])
folderout = fullfile(session.output_path, 'Processed_DATA', ManipName)
BackgroundFile = fullfile(folderout,['Background_cam' num2str(CamNum) '.mat']);

load(BackgroundFile,'BackgroundMin','BackgroundMax','BackgroundMean')
Background=BackgroundMean;

BaseName = join([ManipName '_cam' num2str(CamNum) '_' format],''); % base name of pictures
ImgName = fullfile(folderin,sprintf(BaseName, kframe));
Im = imread(ImgName) - Background;

out=pkfnd(Im,th,sz); % Provides intensity maxima positions
npar = size(out,1);

figure('defaultAxesFontSize',20)
hold on
imagesc(Im)
plot(out(:,1),out(:,2),'or')
%% STEP 5 - CenterFinding2D  
c = clock; fprintf('started at %0.2dh%0.2dm\n',c(4),c(5)) 
% session
firstFrame  = 1;
nframes     = totalnFrames;
th          = allExpeStrct(iexpe).centerFinding_th;
sz          = allExpeStrct(iexpe).centerFinding_sz; 
test        = 0;
%BackgroundType
%format
fprintf('with threshold at %0.0f and size part at %0.0f \n',th,sz) 

CamNum  = 1;
CC1 = CenterFinding2D(session,nameExpe,CamNum,firstFrame,nframes,th,sz); % for camera 1
CamNum  = 2;
CC2 = CenterFinding2D(session,nameExpe,CamNum,firstFrame,nframes,th,sz); % for camera 2
c = clock; fprintf('finished at %0.2dh%0.2dm\n',c(4),c(5)) 

%% STEP 5 - CenterFinding2D  
close all
%% STEP 5 - clean center finding results: here we remove a lot of fake points found 
clear CCtemp
CCtemp = CC2;
%% STEP 5 - filter CC1 and remove all points in the left 
for it = 1 : size(CCtemp,2)
    i2rmv = CCtemp(it).X < 48;
    CCtemp(it).X(i2rmv) = [];
    CCtemp(it).Y(i2rmv) = [];
end
%%  STEP 5 - showing results of center finding
figure
hold on, box on
set(gcf,'position',[0061 0181 1128 0808])
for it = 1 : totalnFrames
    %title(sprintf('time : %0.0f',it))
    plot(CCtemp(it).X,CCtemp(it).Y,'sk')
    %pause(.01)
end
set(gca,'ydir','reverse')
%%  STEP 5 - showing results of center finding - two cameras at same time
c = clock;
sprintf('doing plot at %0.2dh%0.2d',c(4),c(5))
figure
hold on, box on
set(gcf,'position',[0061 0181 0828 0808])
for it = 1 : totalnFrames
    plot(CC1(it).X,CC1(it).Y,'+b')
    plot(CC2(it).X,CC2(it).Y,'+r')
end 

c = clock;
sprintf('done plot at %0.2dh%0.2d',c(4),c(5))








%%  STEP 5 - showing results of center finding - two cameras at same time
c = clock;
sprintf('doing plot at %0.2dh%0.2d',c(4),c(5))
figure
hold on, box on
set(gcf,'position',[0061 0181 0828 0808])
axis([148 158 800 815])
for it = 10: 20 % totalnFrames
    title(sprintf('time : %0.0f',it))
    subplot(2,1,1); hold on; box on
    plot(CC1(it).X,CC1(it).Y,'+b')
    axis([149 156 800 812])
    subplot(2,1,2); hold on; box on
    plot(CC2(it).X,CC2(it).Y,'+r')
    axis([267 273 670 680])
    pause(.5)
end

c = clock;
sprintf('done plot at %0.2dh%0.2d',c(4),c(5))

%% STEP 5 - ui figure that show images from the experiment and the CC displayed over it.

%% STEP 5 - save image sequence with found CC marked on each image
camN = 1;
tic
is =    1;
ie = 4000;
for it = is : 1 : ie
    fprintf('image : %0.0f/%0.0f \n',it,ie)
    
    
    % read image
    cd(strcat(session.input_path,'DATA\',nameExpe,'\',sprintf('cam%0.1d',camN)))
    A = imread(strcat(nameExpe,sprintf('_cam%0.1d_%0.5d.tif',camN,it)));
    ARGB = zeros(size(A,1),size(A,2),3,class(A));
    ARGB(:,:,1) = A; ARGB(:,:,2) = A; ARGB(:,:,3) = A;
    
    
    % add CC on it
    for ip = 1 : length(CCtemp(it).X)
        if isnan(round(CCtemp(it).X(ip))) || isnan(round(CCtemp(it).Y(ip)))
            continue
        end
        for ib = [-5:1:5]
            ARGB(max(min(round(CCtemp(it).Y(ip)) +  5,size(A,1)),1),...
                 max(min(round(CCtemp(it).X(ip)) + ib,size(A,2)),1),3) = 255;
            ARGB(max(min(round(CCtemp(it).Y(ip)) -  5,size(A,1)),1),...
                 max(min(round(CCtemp(it).X(ip)) + ib,size(A,2)),1),3) = 255;
            ARGB(max(min(round(CCtemp(it).Y(ip)) + ib,size(A,1)),1),...
                 max(min(round(CCtemp(it).X(ip)) +  5,size(A,2)),1),3) = 255;
            ARGB(max(min(round(CCtemp(it).Y(ip)) + ib,size(A,1)),1),...
                 max(min(round(CCtemp(it).X(ip)) -  5,size(A,2)),1),3) = 255;
        end
    end
    
    
    % save image
    cd(strcat(session.output_path,'Processed_DATA\',nameExpe,...
        '\checkingCenterFinding2d_imageSequences','\',sprintf('cam%0.1d',camN)))
    imwrite(ARGB,strcat(nameExpe,sprintf('_cam1_%0.5d.png',it)))  
end

toc
c = clock;
fprintf('done at %0.2dh%0.2dm \n',c(4),c(5))


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




%%




