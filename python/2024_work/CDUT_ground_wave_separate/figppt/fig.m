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
clip = 2;
mm = seis(2);

load wudt_STA_res1_p2.mat
load wudt_STA_add_p2.mat
load hyper_in_p2.mat
load windows.mat

zfig(hyper_in_p2,clip,mm);
axis off
zfig(wudt_STA_res1_p2,clip,mm);
axis off
zfig(wudt_STA_add_p2,clip,mm);
axis off

[n1,n2] = size(window);
for i = 1:n2
    for j =1:n1
        if window(j,i) == 0
            window(j,i)=0;
        else
            window(j,i)=1;
        end
    end
end
zfig(window,clip,mm);
axis off
zfig(hyper_in_p2-wudt_STA_res1_p2,clip,mm);
axis off
zfig(window.*wudt_STA_res1_p2,clip,mm);
axis off