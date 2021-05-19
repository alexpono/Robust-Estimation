%% looking for the bug in workflow mcin 2

cd('D:\IFPEN\IFPEN_manips\expe_2021_05_06_calibration\reorderCalPlanes4test')


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
    fprintf('iim: %0.2d, xs: %0.4d, xe: %0.4d, ys: %0.4d, ye: %0.4d, \n',...
             iim,xs,xe,ys,ye)
   A = imread(listNames(iim).name);
   Aminimap(ys:ye,xs:xe) = A(:,:);
end
imshow(Aminimap)
set(gcf,'position',[ 16    48   366   942])
%%
for i = 1 : 1
    
    A = imread(listNames(1).name);
    T = adaptthresh(imgaussfilt(A,2),0.3);
    BW = imbinarize(imgaussfilt(A,2),T);
    imshow(BW), hold on
    set(gcf,'position',[400 48 900 900])
    stats = regionprops(BW,'Centroid','Area');
    clear iKill Xst Yst
    iKill = find([stats.Area] < 500);
    stats(iKill) = [];
    for is = 1 : length(stats)
        Xst(is) = stats(is).Centroid(1,1);
        Yst(is) = stats(is).Centroid(1,2);
        plot(stats(is).Centroid(1,1),stats(is).Centroid(1,2),'or')
    end
    for iBase = 1 : 3
        [x,y] = ginput(1);
        % find the stats point corresponding
        d = sqrt((x-Xst).^2+(y-Yst).^2);
        [a,b] = min(d);
        mirePoints(iBase).e1 = rem(iBase+1,2);
        mirePoints(iBase).e2 = floor((iBase+0)/3);
        mirePoints(iBase).is = b;
        plot(Xst(b),Yst(b),'bo','markerFaceColor','b')
    end
    % use the base to find all other points
    % propagate along e1 axis 
%     while(1)
%         x = 
%         y
%     end
    
    % update minimap
end

%%





