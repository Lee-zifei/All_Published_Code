clear;clc;close all;delete(gcp('nocreate'))  
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
n1 = 500;
n2 = 200;
t1 = 0;
t2 = 50;
x1 = 100;
x2 = 150;
clip = 0.08;
mm = seis(2);
data2 = zread('../d1_newww_curve_result.dat',[n1,n2]);
data1 = zread('../bao9_deb.dat',[n1,n2]);
zfig(data1,clip,mm);
sub_imageplot(data1,lo,t1,t2,x1,x2);