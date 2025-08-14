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
% n1 = 3000;
% n2 = 96;
% n22 = 256;
% 
% clip = 3;
% mm =seis(2);
% 
% patch1 = zread('../test/BGP/patch2.dat',[n1,n22]);
% zfig([patch1],clip,mm);
% 
% 
% input = zread('../test/BGP/input.dat',[n1,n2]);
% result_1 = zread('../test/BGP/result1.dat',[n1,n2]);
% result_2 = zread('../test/BGP/result2.dat',[n1,n2]);
% 
% beg = 80;
% patch1_new = patch1;
% patch1_new(:,beg:beg+n2-1) = -result_2(:,:);
% 
% zfig([input,result_1,result_2],clip,mm);
% set(gca,'FontSize',20,'FontName','Arial');
% ylabel('Time(s)','Fontsize',35,'fontweight','normal');
% xlabel('Trace','Fontsize',35,'fontweight','normal');
% set(gcf,'unit','normalized','position',[0.1,0.1,0.4,0.8]);
% 
% zfig([patch1,patch1_new,(patch1-patch1_new)],clip,mm);
% set(gca,'FontSize',20,'FontName','Arial');
% ylabel('Time(s)','Fontsize',35,'fontweight','normal');
% xlabel('Trace','Fontsize',35,'fontweight','normal');
% set(gcf,'unit','normalized','position',[0.1,0.1,0.4,0.8]);
% 
% zsave('./data_save_compare/patch2/patch2.dat',patch1);
% zsave('./data_save_compare/patch2/patch2_new.dat',patch1_new);
% zsave('./data_save_compare/patch2/patch2_window.dat',input);
% zsave('./data_save_compare/patch2/patch2_result1.dat',result_1);
% zsave('./data_save_compare/patch2/patch2_result2.dat',result_2);

n1 = 3000;
n2 = 96;
n22 = 256;

clip = ;
mm =seis(2);

patch1 = zread('../test/BGP/patch2.dat',[n1,n22]);
zfig([patch1],clip,mm);


input = zread('../test/BGP/input.dat',[n1,n2]);
result_1 = zread('../test/BGP/result1.dat',[n1,n2]);
result_2 = zread('../test/BGP/result2.dat',[n1,n2]);

beg = 80;
patch1_new = patch1;
patch1_new(:,beg:beg+n2-1) = -result_2(:,:);

zfig([input,result_1,result_2],clip,mm);
set(gca,'FontSize',20,'FontName','Arial');
ylabel('Time(s)','Fontsize',35,'fontweight','normal');
xlabel('Trace','Fontsize',35,'fontweight','normal');
set(gcf,'unit','normalized','position',[0.1,0.1,0.4,0.8]);

zfig([patch1,patch1_new,(patch1-patch1_new)],clip,mm);
set(gca,'FontSize',20,'FontName','Arial');
ylabel('Time(s)','Fontsize',35,'fontweight','normal');
xlabel('Trace','Fontsize',35,'fontweight','normal');
set(gcf,'unit','normalized','position',[0.1,0.1,0.4,0.8]);

zsave('./data_save_compare/patch2_all_data_train/patch2.dat',patch1);
zsave('./data_save_compare/patch2_all_data_train/patch2_new.dat',patch1_new);
zsave('./data_save_compare/patch2_all_data_train/patch2_window.dat',input);
zsave('./data_save_compare/patch2_all_data_train/patch2_result1.dat',result_1);
zsave('./data_save_compare/patch2_all_data_train/patch2_result2.dat',result_2);

% xticks(0:10:n2);
% xticklabels({'20','110','200'});
% yticks(0:10:n1);
% yticklabels({'1','2','3','4','5','6','7','8','9'});

