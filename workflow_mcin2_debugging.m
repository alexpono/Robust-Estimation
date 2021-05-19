%% looking for the bug in workflow mcin 2

close all
clear all
name = getenv('COMPUTERNAME');
if strcmp(name,'DESKTOP-3ONLTD9')
    cd(strcat('C:\Users\Lenovo\Jottacloud\RECHERCHE\Projets\21_IFPEN\',...
        'manips\expe_2021_05_06_calibration_COPY\images4calibration'))
elseif strcmp(name,'DARCY')
    cd('D:\IFPEN\IFPEN_manips\expe_2021_05_06_calibration\reorderCalPlanes4test')
end
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

A = imread(listNames(1).name);
[hIm,wIm] = size(A);
hP = figure; % progress in the calibration
Aminimap = zeros(hIm*length(listNames)/2,2*wIm,'uint8');
for iim = 1 : length(listNames)
    xs = 1+rem(iim+1,2)*wIm;
    xe = xs + wIm - 1;
    ys = 1+floor((iim-1)/2)*hIm;
    ye = ys + hIm - 1;
    %fprintf('iim: %0.2d, xs: %0.4d, xe: %0.4d, ys: %0.4d, ye: %0.4d, \n',...
    %         iim,xs,xe,ys,ye)
   A = imread(listNames(iim).name);
   Aminimap(ys:ye,xs:xe) = A(:,:);
end
imshow(Aminimap)
set(gcf,'position',[ 16    48   366   942])

dataAllImages = struct();
for i = 1 : 19
    iloooop = i;
    A = imread(listNames(i).name);
    T = adaptthresh(imgaussfilt(A,1),0.3);
    BW = imbinarize(imgaussfilt(A,2),T);
    hBW = figure; hold on
    imshow(A), hold on
    title(sprintf('plane %0.2d / %0.2d , camera %0.1d',1+floor((i-1)/2),1+rem(i+1,2)))
    set(gcf,'position',[400 48 900 900])
    stats = regionprops(BW,'Centroid','Area','boundingbox','perimeter','convexHull');
    clear iKill Xst Yst
    iKill = find([stats.Area] < 500);
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
        plot(stats(is).Centroid(1,1),stats(is).Centroid(1,2),'+r')
    end

    % find square and triangle
for is = 1 : length(stats)

% same as dCCH but more smooth
poly1 = simplify(polyshape([stats(is).ConvexHull(:,1)],[stats(is).ConvexHull(:,2)],'Simplify',false));
xCC = stats(is).Centroid(1,1);
yCC = stats(is).Centroid(1,2);
for itheta = 1 : 360
    lineseg = [[xCC yCC];...
               [xCC+400*cosd(itheta) yCC+400*sind(itheta)]];
    [in,out] = intersect(poly1,lineseg);
    xCH = in(end,1);
    yCH = in(end,2);
    dCCHBetter(itheta) = sqrt((xCC-xCH)^2+(yCC-yCH)^2);% distance center form to convexhull
end
stats(is).dCCHBetter = dCCHBetter;
stats(is).VdCC = var(dCCHBetter);
[~,~,w,p] = findpeaks(dCCHBetter,[1:1:length(dCCHBetter)],'SortStr','descend');
stats(is).w = w;
stats(is).wsum = sum(w);
stats(is).p = p;
stats(is).psum = sum(p);
end



% criterion for square and triangle

% figure, hold on
% plot([stats.psum],[stats.VdCC],'o')
% xlabel('p')
% ylabel('variance')
% box on

% here I tried automatic finding of the square and the triangle, but it
% does not always work
[~,b] = maxk([stats.VdCC],2);
[~,c] = max( [stats(b(1)).psum , stats(b(2)).psum ] );
iTrgl = b(c);
iSqr  = b(3-c);

% ginput for square and triangle
[xgi,ygi] = ginput(1);
for is = 1 : length(stats)
    Xst(is) = stats(is).Centroid(1,1);
    Yst(is) = stats(is).Centroid(1,2);
end
d = sqrt((xgi-Xst).^2+(ygi-Yst).^2);
[a,b] = min(d);
iTrgl = b;
[xgi,ygi] = ginput(1);
for is = 1 : length(stats)
    Xst(is) = stats(is).Centroid(1,1);
    Yst(is) = stats(is).Centroid(1,2);
end
d = sqrt((xgi-Xst).^2+(ygi-Yst).^2);
[a,b] = min(d);
iSqr = b;

polyTrgl = simplify(polyshape([stats(iTrgl).ConvexHull(:,1)],[stats(iTrgl).ConvexHull(:,2)],'simplify',false));
hpg1 = plot(polyTrgl,'FaceColor',[0.4940 0.1840 0.5560],'FaceAlpha',.5);
polySqr  = simplify(polyshape([stats(iSqr).ConvexHull(:,1)],[stats(iSqr).ConvexHull(:,2)],'simplify',false));
hpg2 = plot(polySqr,'FaceColor',[0 0.4470 0.7410],'FaceAlpha',.5);



% define vectors 
xTg = stats(iTrgl).Centroid(1,1);
yTg = stats(iTrgl).Centroid(1,2);
xSq = stats(iSqr).Centroid(1,1);
ySq = stats(iSqr).Centroid(1,2);

e0505 = [ xTg-xSq ; yTg-ySq ];
theta = 45;
R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
e10 = sqrt(2) * R * e0505;
theta = -45;
R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
e01 = sqrt(2) * R * e0505;

xCC = stats(iSqr).Centroid(1,1);
yCC = stats(iSqr).Centroid(1,2);
x00 = xCC - 0.5 * e10(1);
y00 = yCC - 0.5 * e10(2);

plot(x00,y00,'ob','markerFaceColor','b')
% identify point (0,0)
for is = 1 : length(stats)
    Xst(is) = stats(is).Centroid(1,1);
    Yst(is) = stats(is).Centroid(1,2);
end
d = sqrt((x00-Xst).^2+(y00-Yst).^2);
[a,b00] = min(d);
% refine e10 and e01
d = sqrt((x00+e10(1)-Xst).^2+(y00+e10(2)-Yst).^2);
[a,b10] = min(d);
plot(Xst(b10),Yst(b10),'rs','markerFaceColor','b')
e10 = [Xst(b10)-Xst(b00);Yst(b10)-Yst(b00)];
d = sqrt((x00+e01(1)-Xst).^2+(y00+e01(2)-Yst).^2);
[a,b01] = min(d);
plot(Xst(b01),Yst(b01),'rs','markerFaceColor','b')
e01 = [Xst(b01)-Xst(b00);Yst(b01)-Yst(b00)];

iP = 1;
mirePoints(iP).ist = b00;
mirePoints(iP).xpix = Xst(b00);
mirePoints(iP).ypix = Yst(b00);
mirePoints(iP).xCoord = 0;
mirePoints(iP).yCoord = 0;

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
        plot(Xst,Yst,'ob','markerFaceColor','b')
        iP = iP + 1;
        mirePoints(iP).ist = is;
        mirePoints(iP).xpix = Xst;
        mirePoints(iP).ypix = Yst;
        mirePoints(iP).xCoord = xCoord;
        mirePoints(iP).yCoord = yCoord;
        mirePoints(iP).xCoordEps = xCoordEps;
        mirePoints(iP).yCoordEps = yCoordEps;
    else
        figure(hBW), hold on
        plot(Xst,Yst,'ob','markerFaceColor','r')
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
  
for idai = 1 % index data all images
    
    mirePoints = dataAllImages(idai).mirePoints;
    
end
%%


%%



%%


%%


%%


%%

