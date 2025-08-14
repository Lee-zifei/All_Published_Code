clear;clc;close all  
%Linux
if exist('/media/lzf/Work/code/matlab/mat_toolbox/myprogs','dir')
    addpath('/media/lzf/Work/code/matlab/mat_toolbox/myprogs');
    addpath('/media/lzf/Work/code/matlab/mat_toolbox/CurveLab-2.1.3/fdct_wrapping_matlab');
    addpath(genpath('/media/lzf/Work/code/matlab/mat_toolbox/crewes'));
    datapath='/media/lzf/Work/data'; 
elseif exist('L:\code\matlab\mat_toolbox\myprogs','dir')
%Windows
    addpath('L:\code\matlab\mat_toolbox\myprogs');
    addpath('L:\code\matlab\mat_toolbox\CurveLab-2.1.3\fdct_wrapping_matlab');
    addpath(genpath('L:\code\matlab\mat_toolbox\crewes'));
    datapath='L:\data'; 
elseif exist('/data/data1/lzf/code/matlab/mat_toolbox/myprogs','dir')
%Server
    addpath('/data/data1/lzf/code/matlab/mat_toolbox/myprogs');
    addpath('/data/data1/lzf/code/matlab/mat_toolbox/CurveLab-2.1.3/fdct_wrapping_matlab');
    addpath(genpath('/data/data1/lzf/code/matlab/mat_toolbox/rewes'));
    datapath='/data/data1/lzf/data'; 
else
%MAC
    addpath('/Users/lzf/documents/matlab/Toolbox/myprogs');
    addpath('/Users/lzf/documents/matlab/Toolbox/CurveLab-2.1.3/fdct_wrapping_matlab');
    addpath(genpath('/Users/lzf/documents/matlab/Toolbox/crewes'));
end
%###################################################################################################
n2 = 344;
n1 = 3200;
clip = 10;
mm = seis(2);
dt = 2e-3;

data1 = zread('2ddata/deblend1_1.dat',[n1,n2]);
data2 = zread('2ddata/data21.dat',[n1,n2]);

% data2 = dither1(data2,ones(1,n2)*500);
% zfig(data1,clip,mm);
% zfig(data2,clip,mm);
% zsave('./2ddata/data21.dat',data2)
maintime = 0:6.4:(n2-1)*6.4;
assitime = maintime+2.0 + (3.5-2.0).*rand(1,n2);

blend1 = data1+estimate_interference(data2, assitime,maintime,dt);
blend2_rev = dither1(estimate_interference(blend1, maintime,assitime,dt),-ones(1,n2)*1);
blend2 = data2 + estimate_interference(data1, maintime,assitime,dt);
zfig([data1,blend1,blend1-data1],clip,mm);
zfig([data2,blend2_rev,blend2_rev-data2],clip,mm);
% zfig([blend2_rev,blend2,blend2_rev-blend2,data2],clip,mm);
data22 = estimate_interference(blend1-data1,maintime,assitime,dt);
% % zfig([data2,dither1(data22,-ones(n2)*1),data2-dither1(data22,-ones(n2)*1)],clip,mm);
zsave('data/maintime.dat',maintime);
zsave('data/assitime.dat',assitime);