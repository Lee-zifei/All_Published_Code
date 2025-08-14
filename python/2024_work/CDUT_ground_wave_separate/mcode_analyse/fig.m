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
n1 = 3000;
n2 = 96;
n22 = 256;

clip = 1;
mm =seis(2);

patch1 = zread('./data_save_compare/patch2/patch2.dat',[n1,n22]);
patch1_new = zread('./data_save_compare/patch2/patch2_new.dat',[n1,n22]);
input = zread('./data_save_compare/patch2/patch2_window.dat',[n1,n2]);
result_1 = zread('./data_save_compare/patch2/patch2_result1.dat',[n1,n2]);
result_2 = zread('./data_save_compare/patch2/patch2_result2.dat',[n1,n2]);

zfig([patch1,patch1_new,patch1-patch1_new],clip,mm);

set(gca,'FontSize',20,'FontName','Arial');
ylabel('Time(s)','Fontsize',35,'fontweight','normal');
xlabel('Trace','Fontsize',35,'fontweight','normal');
set(gcf,'unit','normalized','position',[0.1,0.1,0.8,0.8]);

patch1 = zread('./data_save_compare/patch1/patch1.dat',[n1,n22]);
patch1_new = zread('./data_save_compare/patch1/patch1_new.dat',[n1,n22]);
input = zread('./data_save_compare/patch1/patch1_window.dat',[n1,n2]);
result_1 = zread('./data_save_compare/patch1/patch1_result1.dat',[n1,n2]);
result_2 = zread('./data_save_compare/patch1/patch1_result2.dat',[n1,n2]);

zfig([patch1,patch1_new,patch1-patch1_new],clip,mm);

set(gca,'FontSize',20,'FontName','Arial');
ylabel('Time(s)','Fontsize',35,'fontweight','normal');
xlabel('Trace','Fontsize',35,'fontweight','normal');
set(gcf,'unit','normalized','position',[0.1,0.1,0.8,0.8]);

patch1 = zread('./data_save_compare/patch2_all_data_train/patch2.dat',[n1,n22]);
patch1_new = zread('./data_save_compare/patch2_all_data_train/patch2_new.dat',[n1,n22]);
input = zread('./data_save_compare/patch2_all_data_train/patch2_window.dat',[n1,n2]);
result_1 = zread('./data_save_compare/patch2_all_data_train/patch2_result1.dat',[n1,n2]);
result_2 = zread('./data_save_compare/patch2_all_data_train/patch2_result2.dat',[n1,n2]);

zfig([patch1,patch1_new,patch1-patch1_new],clip,mm);

set(gca,'FontSize',20,'FontName','Arial');
ylabel('Time(s)','Fontsize',35,'fontweight','normal');
xlabel('Trace','Fontsize',35,'fontweight','normal');
set(gcf,'unit','normalized','position',[0.1,0.1,0.8,0.8]);

