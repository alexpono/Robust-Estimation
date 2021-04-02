function [traj,tracks]=TAN_track2d(pos,maxdist,longmin)

% nearest neighbor particle tracking algo
% pos is a structure with field .pos : pos.pos(:,1)=x1 ; pos.pos(:,2)=x2
% pos.pos(:,3)=framenumber
% frame number must an integer from 1->N
% maxdist=1; disc radius in where we are supposed to find a particle in
% next frame
% longmin : minimum length trajectory
%
% in case there are few frames bit many particles this code could be
% improved using say 100^2 vertex (only try particle in nearest vertices)
%
% example:
% load tracking_rom_positions.mat
% tracks=track2d_rom(positions,1,50);
%
% use plot_traj to display results
%
% written by r. volk 09/2014 (modified 01/2020)

tic;

tracks = pos;
for ii = 1:size(tracks,2)
    % frame number in the trajectory (from 1 ->p) for a trajectory of length p
    tracks(ii).pos(:,4) = zeros(size(tracks(ii).pos,1),1);
    % 5th column 1 if particle is free to be linked to a new trajectory, 0 if no, 2 if
    % linked to 2 or more trajectories.
    tracks(ii).pos(:,5) = ones(size(tracks(ii).pos,1),1);
    % we don't create trajectories, we only write numbers in the 4th and 5th
    % columns
end


% number of active tracks, we strat with frame 1
ind_actif = (1:size(tracks(1).pos,1));

tracks(1).pos(:,4) = (1:size(tracks(1).pos,1));
tracks(1).pos(:,5) = zeros(size(tracks(1).pos,1),1);

% number of trajectories created at this step
% will increase each time we create a new trajectory
numtraj=size(tracks(1).pos,1);

% loop over frames
%tic
for kk=2:size(tracks,2)
    % frame number we are looking at
    %numframe=kk;
    % indices of those paricles
    %ind_new=find(tracks(:,3)==numframe);

    % loop over active particles in previous frame (kk-1)
    for ll=1:length(ind_actif)
        % position of particle ll in frame kk-1
        actx = tracks(kk-1).pos(ind_actif(ll),1);
        acty = tracks(kk-1).pos(ind_actif(ll),2);

        % trajectory number of the active particle (frame kk-1)
        actnum = tracks(kk-1).pos(ind_actif(ll),4);

        % could add a tag: frame number in this trajectory
        % si tag<=2 : rien
        % si tag==3 actx=actx+vx*dt avec dt=1 ici (3 frames best estimate)
        % si tag==4 actx

        % new particle positions in frame kk
        newx = tracks(kk).pos(:,1);
        newy = tracks(kk).pos(:,2);


        % compute distance
        dist = sqrt((actx-newx).^2+(acty-newy).^2);
        % take the min
        [dmin,ind_min]=min(dist);

        % test with maxdist criterion
        if dmin < maxdist
            dispo = tracks(kk).pos(ind_min,5);

            if dispo==1
                % part is dispo=free we change dispo into 0
                tracks(kk).pos(ind_min,5) = 0;

                % we link the particle to the active particle set
                % trajectory number equal to the one of the active
                % particle
                tracks(kk).pos(ind_min,4) = actnum;

            elseif dispo==0
                % the part already linked, change dispo into 2
                % can't be linked to 2 trajectories
                tracks(kk).pos(ind_min,5) = 2;

                % and we set its trajectory number to zero
                % will be rejected at the end
                tracks(kk).pos(ind_min,4) = 0;

            end
        end
    end

        % define particles to be tracked
        % keep particles found only one time, and the non found particles
        % those will create new trajectories
        ind_actif = find(tracks(kk).pos(:,5)==0);

        % new (not found) particles are given a new trajectory number
        ind_new_part = find(tracks(kk).pos(:,5)==1);

        % if there are new particles
        if isempty(ind_new_part)==0
            % loop of new part -> increase numtraj
            for mm=1:length(ind_new_part)
                numtraj = numtraj + 1;
                tracks(kk).pos(ind_new_part(mm),4) = numtraj;
            end
        end
        ind_actif=[ind_actif;ind_new_part];
    %toc
end

% write trajectories in right order.
% reject all particles found 2 times
% keep only trajectories longer than longmin

tracks_array = cat(1,tracks.pos); %put all traj in an array
tracks_array = sortrows(tracks_array,4); %sort traj in ascendent order

tracks_array = tracks_array(tracks_array(:,4)~=0,:); %kick numtraj == 0 
fstart = zeros(length(tracks_array),1);
fstart(1) = 1; fstart(2:end) = diff(tracks_array(:,4)); %make diff -> 0 when two 2 succesive lines belong to the same traj, 1 if not
fstart = find(fstart==1); %find indices corresponding to the start of trajectories

flong = diff(fstart); %find the length of trajectories (except the last one)
last_traj = find(tracks_array(:,4)==length(fstart)); %find the length of the last trajectory
flong(end+1) = length(last_traj);

ftrackselec = find(flong>=longmin); %select traj longer than longmin

traj = struct(); %rearange in a structure
for kk = 1 :length(ftrackselec)
    ll  = flong(ftrackselec(kk));
    deb = fstart(ftrackselec(kk));
    traj(kk).track = tracks_array(deb:deb+ll-1,:);
end
toc