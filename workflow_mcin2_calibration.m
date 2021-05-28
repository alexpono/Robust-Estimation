%% looking for the bug in workflow mcin 2

close all
clear all
name = getenv('COMPUTERNAME');
if strcmp(name,'DESKTOP-3ONLTD9')
    cd(strcat('C:\Users\Lenovo\Jottacloud\RECHERCHE\Projets\21_IFPEN\',...
        'manips\expe_2021_05_06_calibration_COPY\images4calibration'))
elseif strcmp(name,'DARCY')
    % cd('D:\IFPEN\IFPEN_manips\expe_2021_05_06_calibration\reorderCalPlanes4test')
    cd('D:\IFPEN\IFPEN_manips\expe_2021_05_28_calibration_in_air\calibration02\forCalibration')
    % cd('C:\Users\darcy\Desktop\git\Robust-Estimation\calibrationImagesTraining01')
end

zPlane = [00:10:40]; % [00:05:40]; % mm
camName = {1,2};
gridSpace = 5;        % mm

%% modify the names of the calibration pictures

listNames = dir('*.tif');

nPlans = length(listNames)/2;
for il = 1 : nPlans
    fprintf('plan:%0.2d \n',il)
    
    ic1 = 2*il-1;
    nameIn  = listNames(ic1).name;
    nameOut = strcat( listNames(ic1).name(1:16),...
        sprintf('%0.2d',nPlans-(il-1)),...
        listNames(ic1).name(18:end));
    fprintf('%s becomes %s \n',nameIn,nameOut)
    A = imread(nameIn);
    imwrite(A,nameOut)
    
    ic2 = 2*il;
    nameIn  = listNames(ic2).name;
    nameOut = strcat( listNames(ic2).name(1:16),...
        sprintf('%0.2d',nPlans-(il-1)),...
        listNames(ic2).name(18:end));
    fprintf('%s becomes %s \n',nameIn,nameOut)
    A = imread(nameIn);
    imwrite(A,nameOut)
    
end
%% 
close all
clear listNames mirePoints
mirePoints = struct();
listNames = dir('*.tif');

% automatic detection of number of cameras and number of planes
nPlanes  = length(zPlane);
nCameras = length(camName);

A = imread(listNames(1).name);
[hIm,wIm] = size(A);
classImages = class(A);

hP = figure; % progress in the calibration
Aminimap = zeros(hIm*length(listNames)/nCameras,nCameras*wIm,classImages);
for iim = 1 : length(listNames)
    xs = 1+rem(iim-1,nCameras)*wIm;
    xe = xs + wIm - 1;
    ys = 1+floor((iim-1)/nCameras)*hIm;
    ye = ys + hIm - 1;
    fprintf('iim: %0.2d, xs: %0.4d, xe: %0.4d, ys: %0.4d, ye: %0.4d, \n',...
             iim,xs,xe,ys,ye)
   A = imread(listNames(iim).name);
   Aminimap(ys:ye,xs:xe) = A(:,:);
end
imshow(Aminimap)
set(gcf,'position',[ 16    48   366   942])


dataAllImages = struct();
for i = 1 : length(listNames)
    clear  mirePoints
    mirePoints = struct();
    iloooop = i;
    A = imread(listNames(i).name);
    T = adaptthresh(imgaussfilt(A,1),0.3);
    BW = imbinarize(imgaussfilt(A,2),T);
    hBW = figure; hold on
    imagesc(256-A), colormap gray, hold on, box on
    axis([0 hIm 0 wIm])
    set(gca,'ydir','reverse')
    title(sprintf('triangle then square -- plane %0.2d / 09 , camera %0.1d',...
        1+floor((i-1)/2),1+rem(i+1,2)))
    set(gcf,'position',[400 48 900 900])
    stats = regionprops(BW,'Centroid','Area','boundingbox','perimeter','convexHull');
    clear iKill Xst Yst
    iKill = find([stats.Area] < 100);
    stats(iKill) = [];
    clear iKill
    iKill = find([stats.Area] > 1700);
    stats(iKill) = [];
    clear iKill
    iKill = [];
    for is = 1 : length(stats)
        as = stats(is).BoundingBox(3)/stats(is).BoundingBox(4); % aspect ratio
        if  as > 2 || as < 0.5
            iKill = [iKill,is];
        end
    end
    stats(iKill) = [];
    clear iKill
    iKill = [];
    for is = 1 : length(stats)
        Xc = stats(is).Centroid(1,1);
        Yc = stats(is).Centroid(1,2);
        if (Xc < 50) || (Xc > (wIm-50)) || (Yc < 50) || (Yc > (hIm-50))
            iKill = [iKill,is];
        end
    end
    stats(iKill) = [];

    for is = 1 : length(stats)
        Xst(is) = stats(is).Centroid(1,1);
        Yst(is) = stats(is).Centroid(1,2);
        %plot(stats(is).Centroid(1,1),stats(is).Centroid(1,2),'+r')
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % find square and triangle

% try to automatic finding the triangle and square
clear Xst Yst
for is = 1 : length(stats)
    Xst(is) = stats(is).Centroid(1,1);
    Yst(is) = stats(is).Centroid(1,2);
end
% find the two guys who have the four closest neighbourgs
ddd = sqrt((Xst'-Xst).^2+(Yst'-Yst).^2);
ii=ones(size(ddd));
%idx=triu(ii,1);
%ddd(~idx)=nan;
idx = logical( diag(ones(size(ddd,1),1)) );
ddd(~(~idx)) = nan;
B = sum(mink(ddd,4));
[~,b] = mink(B,2);
if stats(b(1)).Area < stats(b(2)).Area
    iTrgl = b(1);
    iSqr  = b(2);
else
    iTrgl = b(2);
    iSqr  = b(1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
polyTrgl = simplify(polyshape([stats(iTrgl).ConvexHull(:,1)],[stats(iTrgl).ConvexHull(:,2)],'simplify',false));
hpg1 = plot(polyTrgl,'FaceColor',[0.4940 0.1840 0.5560],'FaceAlpha',.5);
polySqr  = simplify(polyshape([stats(iSqr).ConvexHull(:,1)],[stats(iSqr).ConvexHull(:,2)],'simplify',false));
hpg2 = plot(polySqr,'FaceColor',[0 0.4470 0.7410],'FaceAlpha',.5);


% define vectors 
xTg = stats(iTrgl).Centroid(1,1);
yTg = stats(iTrgl).Centroid(1,2);
xSq = stats(iSqr).Centroid(1,1);
ySq = stats(iSqr).Centroid(1,2);

% FACE A
e0505 = [ -xTg+xSq ; -yTg+ySq ];%[ xTg-xSq ; yTg-ySq ];
 theta = -45;
% FACE B
%e0505 = [ xTg-xSq ; yTg-ySq ];
%theta = 45;

R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
e10 = sqrt(2) * R * e0505;
% FACE A
theta = -45-90;
% FACE B
% theta = 45-90;
R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
e01 = sqrt(2) * R * e0505;

xCC = stats(iTrgl).Centroid(1,1);
yCC = stats(iTrgl).Centroid(1,2);
x00 = xCC - 0.5 * e10(1);
% FACE A
y00 = yCC - 0.0 * e10(2);
% FACE B
% y00 = yCC - 0.5 * e10(2);

plot(x00,y00,'ob','markerFaceColor','b')
text(x00,y00,'(0,0)')
% identify point (0,0)
for is = 1 : length(stats)
    Xst(is) = stats(is).Centroid(1,1);
    Yst(is) = stats(is).Centroid(1,2);
end
d = sqrt((x00-Xst).^2+(y00-Yst).^2);
[a,b00] = min(d);
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%
% refine e10 and e01
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%
d = sqrt((x00+e10(1)-Xst).^2+(y00+e10(2)-Yst).^2);
[a,b10] = min(d);
% plot(Xst(b10),Yst(b10),'rs','markerFaceColor','b')
e10 = [Xst(b10)-Xst(b00);Yst(b10)-Yst(b00)];
d = sqrt((x00+e01(1)-Xst).^2+(y00+e01(2)-Yst).^2);
[a,b01] = min(d);
% plot(Xst(b01),Yst(b01),'rs','markerFaceColor','b')
e01 = [Xst(b01)-Xst(b00);Yst(b01)-Yst(b00)];
quiver(Xst(b00),Yst(b00),e10(1),e10(2),'Color','b','LineWidth',3)
quiver(Xst(b00),Yst(b00),e01(1),e01(2),'Color','r','LineWidth',3)

iP = 0;
% normed versions of the vectors
e10n = e10/(norm(e10)^2);
e01n = e01/(norm(e01)^2);
% loop on all the stats points and identify them
for is = 1 : length(stats)
    clear Xst Yst 
    Xst = stats(is).Centroid(1,1);
    Yst = stats(is).Centroid(1,2);
    UV = [Xst-x00;Yst-y00];
    
    UVx = dot( e10n, UV);
    xCoord    = round( UVx );
    xCoordEps = abs(UVx - xCoord);
    UVy = dot( e01n, UV);
    yCoord    = round( UVy );
    yCoordEps = abs(UVy - yCoord);
    
    
    if xCoordEps + yCoordEps < 0.25
        figure(hBW), hold on
        %plot(Xst,Yst,'ob','markerFaceColor','b')
        iP = iP + 1;
        mirePoints(iP).ist = is;
        mirePoints(iP).xpix = Xst;
        mirePoints(iP).ypix = Yst;
        mirePoints(iP).xCoord = xCoord;
        mirePoints(iP).yCoord = yCoord;
        mirePoints(iP).xCoordEps = xCoordEps;
        mirePoints(iP).yCoordEps = yCoordEps;
        clear polyShp
        polyShp = simplify(polyshape([stats(is).ConvexHull(:,1)],[stats(is).ConvexHull(:,2)],'simplify',false));
        hpg1 = plot(polyShp,'FaceColor',[0.1 0.1 0.7],'FaceAlpha',.5);
        text(Xst+20,Yst+20,sprintf('(%0.0f,%0.0f)',xCoord,yCoord))
        mirePoints(iP).ConvexHull(:,1) = stats(is).ConvexHull(:,1);
        mirePoints(iP).ConvexHull(:,2) = stats(is).ConvexHull(:,2);
    else
        %figure(hBW), hold on
        %plot(Xst,Yst,'ob','markerFaceColor','r')

    end 
end

dataAllImages(i).mirePoints = mirePoints;

    % update minimap
    figure(hP),hold on
    xs = 1+rem(iloooop+1,2)*wIm;
    xe = xs + wIm - 1;
    ys = 1+floor((iloooop-1)/2)*hIm;
    ye = ys + hIm - 1;
    patch('xdata',[xs,xe,xe,xs],'ydata',[ys,ys,ye,ye],...
        'faceAlpha',.3,'faceColor',[0.2 0.2 0.8])
end



%% build the calib file

savepath = 'D:\IFPEN\IFPEN_manips\expe_2021_05_28_calibration_in_air\calibrationFile';
%savepath = 'D:\IFPEN\IFPEN_manips\expe_2021_05_20_calibration_air\forCalib';
% from calib2D function
% pimg      : center coordinates in original image [in pixels]
% pos3D     : center coordinates in real world [in mm]
% T3rw2px   : transformation from real world to image cubic
% T1rw2px   : transformation from real world to image linear
% T3px2rw   : transformation from image to real world cubic
%
% from MakeCalibration function
%     calib(kz,kcam).posPlane        : z position which corresponds to kz plane
%     calib(kz,kcam).pimg            : detected mire points on the calibration picture (2D) in px units,
%     calib(kz,kcam).pos3D           : detected mire points in 3D space and in SI units,
%     calib(kz,kcam).movedpoints     : index of moved points,
%     calib(kz,kcam).addedpoints     : index of added points,
%     calib(kz,kcam).T1rx2px         : Linear transformation from real
%     world to px. T1rw2px=inverse(T1px2rw), Not saved as only one of the
%     two is sufficient.
%     calib(kz,kcam).T3rw2px         : Cubic transformation from real world to px. Pay attention T3rw2px~=inverse(T3px2rw) !,
%     calib(kz,kcam).T1px2rw         : Linear transformation from px to real world,
%     calib(kz,kcam).T3px2rw         : Cubic transformation from px to real world,
%     calib(kz,kcam).cHull           : coordinates of the region of interest,
%     calib(kz,kcam).name            : camera number (kcam),
%     calib(kz,kcam).dirPlane        : [i,j,k] axes orientation. i,j = axis provided by the calibration mire, k is the axis along which the mire is displaced. For instance, if the mire is in the plane (Oxy) and that it is displaced along z axis, dirPlane=[1,2,3]. 

for idai = 1 : length(dataAllImages)
    clear kz camNumber mirePoints PNpimg xyCoord PNpos3D PNpos2D
    kz  = 1+floor((idai-1)/2);
    camNumber = 1+rem(idai+1,2);
    fprintf('idai: %0.2d, kz: %0.2d, cam n°: %0.2d \n',idai,kz,camNumber)
    % build PNpimg
    mirePoints = dataAllImages(idai).mirePoints;
    PNpimg(1:length(mirePoints),1) = [mirePoints.xpix];
    PNpimg(1:length(mirePoints),2) = [mirePoints.ypix];
    PNpimg = sortrows(PNpimg,1);
    
    % build PNpos3D
    % sort xCoord and yCoord
    xyCoord(:,1) = [mirePoints.xCoord];
    xyCoord(:,2) = [mirePoints.yCoord];
    xyCoord = sortrows(xyCoord,1);
    for ixy = 1 : length(xyCoord)
        PNpos3D(ixy,1) = gridSpace * xyCoord(ixy,1);
        PNpos3D(ixy,2) = gridSpace * xyCoord(ixy,2);
        PNpos3D(ixy,3) = zPlane(kz);
    end
    PNpos2D      = PNpos3D(:,1:2);         % position of dots in 2d [mm]
    
    clear T3rw2px T3px2rw T1rw2px pos3D pimg movedpoints addedpoints convHullpimg
    % compute 3rd order polynomial spatial transformation from image points
    % [in pixels] to 2d position in real-space on plane [in mm]
    ttype = 'polynomial';
    T3rw2px  = fitgeotrans(PNpimg,PNpos2D,ttype,3); % inverse transform
    T3px2rw  = fitgeotrans(PNpos2D,PNpimg,ttype,3); % forward transform
    
    ttype = 'projective';
    T1rw2px  = fitgeotrans(PNpimg,PNpos2D,ttype); % inverse transform
    
    pos3D = PNpos3D;
    pimg  = PNpimg;
    movedpoints = zeros(size(pos3D,1),1);
    addedpoints = zeros(size(pos3D,1),1);
    % save results to file
    convHullpimg = convhull(PNpimg(:,1),PNpimg(:,2));
    save(sprintf('%s/calib2D_%d_cam%d',savepath, kz ,camNumber),...
        'zPlane','T3rw2px','T3px2rw','T1rw2px','pos3D','pimg','convHullpimg','movedpoints','addedpoints');

end
%% make calib structure
Ncam = 2;
for kz = 1:numel(zPlane)
    for kcam = 1:Ncam
        load([savepath filesep 'calib2D_' num2str(kz) '_cam' num2str(kcam) '.mat']);
        calib(kz,kcam).posPlane = zPlane(kz);
        calib(kz,kcam).pimg = pimg;
        calib(kz,kcam).pos3D = pos3D;
        calib(kz,kcam).movedpoints = movedpoints;
        calib(kz,kcam).addedpoints = addedpoints;
%         calib(kz,kcam).T1rw2px = fitgeotrans(pimg,pos3D(:,1:2),'projective');
        calib(kz,kcam).T3rw2px = fitgeotrans(pimg,pos3D(:,1:2),'polynomial',3);
        calib(kz,kcam).T1px2rw = fitgeotrans(pos3D(:,1:2),pimg,'projective');
        calib(kz,kcam).T3px2rw = fitgeotrans(pos3D(:,1:2),pimg,'polynomial',3);
        calib(kz,kcam).cHull=convHullpimg;
        calib(kz,kcam).name = kcam;
        calib(kz,kcam).dirPlane=[1,2,3]; % Mire is displaced along axis 3, and 1,2 because the transformation provide x and y in real world

    end
end

save(sprintf('%s/calib.mat',savepath),'calib');



%%


%%


%%

