function [ Gr, effabs ] = GradientFromSolver_1D( img, wavelength, desired_angle)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes he


img = img/2.0 + 0.5;

min_feat = 50;
n_top = 1;
% n_low = n_low;
n_sub = 1.45;
n_in = 1;

load('p_Si.mat');
n_Si = interp1(WL,n,wavelength);

 

layers = 1;
z_step = min(wavelength)/40; % z step for z discritization

thickness = 325;
incident_angle = 0;
desired_angle = desired_angle; % deg, in air
min_feat = 60;
topvar = 1;

pol = {'TM'};
period = wavelength/(sind(desired_angle));


% forward simulation
k_par_f = sind(incident_angle)*n_sub; % incident moment
nn = ceil(12*period/min(wavelength));

retio([],inf*1i);

if strcmp(pol,'TE')
    parm = res0(1);
elseif strcmp(pol,'TM')
    parm = res0(-1);
end
parm.res1.champ = 1;
num_z_disc = round(thickness/z_step);
parm.res3.npts = [0,num_z_disc, 0];

% define textures
textures = cell(1,3);
textures{1} = {n_top};
textures{2} = {n_sub};

nlength = length(img);
dx = period/nlength;
xvec = [1:nlength]*dx - 0.5*period;
nvec = img*(n_Si - n_in) + n_in;

textures{3} = {xvec,nvec};


% define profile
profile = {[0,thickness,0],[1,3,2]};


parm.res3.sens = -1;
parm.res1.champ = 1;
aa = res1(wavelength,period,textures,nn,k_par_f,parm);
result1 = res2(aa,profile);
[et,z,index] = res3(xvec,aa,profile,1,parm);

% Getting transmitted orders and efficiencies
tr = result1.inc_bottom_transmitted;
tgtcur = 1;
[thetadiff,ind_target] = min(abs(tr.order-tgtcur));
effabs = tr.efficiency(ind_target);            % to save

            
% backward
k_par_r = -tr.K(ind_target,1);
aa = res1(wavelength,period,textures,nn,k_par_r,parm);
result2 = res2(aa,profile);
tr_r = result2.inc_top_transmitted;
[thetadiff,ind0r] = min(abs(tr_r.theta));
parm.res1.champ = 1;
parm.res3.sens = 1;

%     FOM gradient


stp = 0.07;


if strcmp(pol,'TE')
    [er,z,index] = res3(xvec,aa,profile,exp(1i*angle(conj(tr.E(ind_target,2)))),parm);
    gr0all = real(-1i*(et(:,:,1).*er(:,:,1)));

elseif strcmp(pol,'TM')
    [er,z,index] = res3(xvec,aa,profile,exp(1i*angle(conj(tr.H(ind_target,2)))),parm);
    EEpar = et(:,:,3).*er(:,:,3);
    EEnorm = et(:,:,2).*er(:,:,2);
    gr0all = real(1i*(EEpar+EEnorm));
end

grall = mean(gr0all, 1);
grtot = grall;
grtot(1:round(min_feat/2)) = 0;
grtot((nlength-round(min_feat/2)):nlength) = 0;
grtot((img==1)&(grtot>0)) = 0;
grtot((img==0)&(grtot<0)) = 0;

grtot = grtot/max(abs(grtot));

%grtot = stp*grtot;
%grtot((grtot+img)>1) = 1-img((grtot+img)>1);
%grtot((img+grtot)<0) = -img((img+grtot)<0);
%grtot = grtot/(max(abs(grtot));
%%grtot = stp*grtot;
%grtot((a<0)&(grtot+a>0)) = -a((a<0)&(grtot+a>0));
%grtot((a>1)&(grtot+a<1)) = 1 - a((a>1)&(grtot+a<1));
%grtot = grtot/max(max(abs(grtot)));
%grtot = stp*grtot;


Gr = grtot*2.0;    

end

