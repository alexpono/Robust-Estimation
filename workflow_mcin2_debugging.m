%% looking for the bug in workflow mcin 2


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
[h,w] = size(A);
hP = figure; % progress in the calibration
Aminimap = zeros(h*length(listNames)/2,2*w,'uint8');
for iim = 1 : length(listNames)
    xs = 1+rem(iim+1,2)*w;
    xe = xs + w - 1;
    ys = 1+floor((iim-1)/2)*h;
    ye = ys + h - 1;
    %fprintf('iim: %0.2d, xs: %0.4d, xe: %0.4d, ys: %0.4d, ye: %0.4d, \n',...
    %         iim,xs,xe,ys,ye)
   A = imread(listNames(iim).name);
   Aminimap(ys:ye,xs:xe) = A(:,:);
end
imshow(Aminimap)
set(gcf,'position',[ 16    48   366   942])

A = imread(listNames(1).name);
[h,w] = size(A);
for i = 1 : 1
    
    A = imread(listNames(1).name);
    T = adaptthresh(imgaussfilt(A,1),0.3);
    BW = imbinarize(imgaussfilt(A,2),T);
    hBW = figure; hold on
    imshow(BW), hold on
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
        if Xc < 50 || Xc > w-50 || Yc < 50 || Yc > h-50
            iKill = [iKill,is];
        end
    end
    stats(iKill) = [];

    for is = 1 : length(stats)
        Xst(is) = stats(is).Centroid(1,1);
        Yst(is) = stats(is).Centroid(1,2);
        plot(stats(is).Centroid(1,1),stats(is).Centroid(1,2),'+r')
    end

    
    % update minimap
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


%%
% criterion for square and triangle

% figure, hold on
% plot([stats.psum],[stats.VdCC],'o')
% xlabel('p')
% ylabel('variance')
% box on

[~,b] = maxk([stats.VdCC],2);
[~,c] = max( [stats(b(1)).psum , stats(b(2)).psum ] );
iTrgl = b(c);
iSqr  = b(3-c);
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
[a,b] = min(d);

iP = 1;
mirePoints(iP).ist = b;
mirePoints(iP).xpix = Xst(b);
mirePoints(iP).ypix = Yst(b);
mirePoints(iP).xCoord = 0;
mirePoints(iP).yCoord = 0;

% goes up
while 1
    x00 = Xst(b) + 0 * e01(1);
    y00 = Yst(b) + 1 * e01(2);
    for is = 1 : length(stats)
        Xst(is) = stats(is).Centroid(1,1);
        Yst(is) = stats(is).Centroid(1,2);
    end
    d = sqrt((x00-Xst).^2+(y00-Yst).^2);
    [a,b] = min(d);
    if a > 20
        break
    end
    iP = iP + 1;
    mirePoints(iP).ist = b;
    mirePoints(iP).xpix = Xst(b);
    mirePoints(iP).ypix = Yst(b);
    mirePoints(iP).xCoord = 0;
    mirePoints(iP).yCoord = mirePoints(iP-1).yCoord + 1;
    plot(Xst(b),Yst(b),'ob','markerFaceColor','b')
end

% goes down
b = find(([mirePoints.xCoord]==0) == ([mirePoints.yCoord]==0));
b = mirePoints(b).ist;
yCoord = 0;

while 1
    yCoord = yCoord - 1;
    % restart from point (0,0)
    x00 = Xst(b) + 0 * e01(1);
    y00 = Yst(b) - 1 * e01(2);
    for is = 1 : length(stats)
        Xst(is) = stats(is).Centroid(1,1);
        Yst(is) = stats(is).Centroid(1,2);
    end
    d = sqrt((x00-Xst).^2+(y00-Yst).^2);
    [a,b] = min(d);
    if a > 20
        break
    end
    iP = iP + 1;
    mirePoints(iP).ist = b;
    mirePoints(iP).xpix = Xst(b);
    mirePoints(iP).ypix = Yst(b);
    mirePoints(iP).xCoord = 0;
    mirePoints(iP).yCoord = yCoord;
    plot(Xst(b),Yst(b),'ob','markerFaceColor','b')
end

%%


%%


%% OLD 
stW(is) = stats(is).BoundingBox(1,3);
stH(is) = stats(is).BoundingBox(1,4);
stP1(is) = 2*stW(is)+2*stH(is);
stP2(is) = stats(is).Perimeter;
xCC = stats(is).Centroid(1,1);
yCC = stats(is).Centroid(1,2);
for ich = 1 : length(stats(is).ConvexHull)
    xCH = stats(is).ConvexHull(ich,1);
    yCH = stats(is).ConvexHull(ich,2);
    dCCH(ich) = sqrt((xCC-xCH)^2+(yCC-yCH)^2);% distance center form to convexhull
end
stats(is).dCCH = dCCH;


%%


%%


%%


%%

