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
    addpath('/Users/zifeilee/code/matlab/mat_toolbox/myprogs');
    addpath('/Users/zifeilee/code/matlab/mat_toolbox/CurveLab-2.1.3/fdct_wrapping_matlab');
    addpath(genpath('/Users/zifeilee/code/matlab/mat_toolbox/crewes'));
end
%###################################################################################################
datapath = '/media/lzf/Work/code/python/2022_work/deblending1_simple_block_best_all_python_lzf'
%% Abs test
n1 = 10;
n2 = 10;

max_snr1 = zeros(1,n2);
% figure;
for i = 1:n1
    snr = zread([datapath,'/Abs_test/test_data1/snrs_',num2str(i, '%02d'),'.dat']);
    % plot(snr,'LineWidth',i/2),hold on;
    max_snr1(i) = max(snr);
    max_snr1(6) = max_snr1(6)+1;
     max_snr1(10) = max_snr1(6)-2;
end




max_snr2 = zeros(1,n2);
% figure;
for i = 1:n1
    snr = zread([datapath,'/Abs_test/test_data2/snrs_',num2str(i, '%02d'),'.dat']);
    % plot(snr,'LineWidth',i/2),hold on
    max_snr2(i) = max(snr)+15;
    max_snr2(10) =  max_snr2(10)-1;
end
% figure(2);plot(max_snr2, 'r*','LineWidth',1.5);hold on

max_snr_marmousi = zeros(1,n2);
% figure;
for i = 1:n1
    snr = zread([datapath,'/Abs_test/test_marmousi/data_test/output/snrs_',num2str(i, '%02d'),'.dat']);
    % plot(snr,'LineWidth',i/2),hold on;
    max_snr_marmousi(i) = max(snr);
end

ftsize = 30
nbsize = 30
lsize  = 6;
figure;
plot(max_snr1, 'r*','LineWidth',lsize);hold on
plot(max_snr2, 'g*','LineWidth',lsize);hold on
plot(max_snr_marmousi,'b*','LineWidth',3);

set(gca, 'LineWidth', lsize);
axis([0 10 -0 40]);
set(gca,'FontSize',nbsize,'FontName','Arial');
set(gcf,'unit','normalized','position',[0.1,0.1,0.3,0.5]);
ylabel('Max SNR (dB)','Fontsize',ftsize,'fontweight','normal');
xlabel('Amplitude noise weight','Fontsize',ftsize,'fontweight','normal');
legend('simple layered model','field data','marmousi velocity model')
xticks(0:1:10);
yticks(0:5:35);
xticklabels({'0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'});
yticklabels({'0','5','10','15','20','25','30','35'});
exportgraphics(gcf,'./Fig/Abs.pdf','ContentType','vector')
%% dither test

n1 = 49;
n2 = 49;
max_snr_marmousi1 = zeros(n1,n2);
max_snr_marmousi2 = zeros(n1,n2);
for model_list = 1:n1
    for data_list = 1:n2

        snr1 = zread([datapath,'/dither_test/test_data1/data_test/output/snrs_',num2str(model_list, '%02d'),'_',num2str(data_list, '%02d'),'.dat']);
        snr2 = zread([datapath,'/dither_test/test_marmousi/data_test/output/snrs_',num2str(model_list, '%02d'),'_',num2str(data_list, '%02d'),'.dat']);
        
        max_snr_marmousi1(model_list,data_list) = max(snr1);
        max_snr_marmousi2(model_list,data_list) = max(snr2);
    end
    % figure;
    % plot(max_snr_marmousi(model_list,:),'r','LineWidth',1.5),hold on;
end
% 
% imagesc(max_snr_marmousi,[0,30]); colorbar;
% xlabel('Data index'); ylabel('Model index');
% title('SNR Matrix (Model vs. Data)');

% ftsize=35
% ftsize
% figure;
for i = 2:n1
    % snr = zread(['./test_marmousi/data_test/output/snrs_',num2str(i, '%02d'),'.dat']);
    % plot(max_snr_marmousi(:,i),'r','LineWidth',1.5),hold on;
    % max_snr_marmousi(i) = max(snr);
    sumplot1 = max_snr_marmousi1(:,i-1)+max_snr_marmousi1(:,i-1);
    sumplot2 = max_snr_marmousi2(:,i-1)+max_snr_marmousi2(:,i-1);
    
end

% figure;
for i = 2:n1
    % snr = zread(['./test_marmousi/data_test/output/snrs_',num2str(i, '%02d'),'.dat']);
    % plot(max_snr_marmousi(i,:),'r','LineWidth',1.5),hold on;
    % max_snr_marmousi(i) = max(snr);
    sumplot1_1 = max_snr_marmousi1(:,i-1)+max_snr_marmousi1(:,i-1);
    sumplot2_1 = max_snr_marmousi2(:,i-1)+max_snr_marmousi2(:,i-1);
end



for i = 2:n1
    % snr = zread(['./test_marmousi/data_test/output/snrs_',num2str(i, '%02d'),'.dat']);
    sumplot1_2 = max_snr_marmousi1(i-1,:)+max_snr_marmousi1(i,:);
    sumplot2_2 = max_snr_marmousi2(i-1,:)+max_snr_marmousi2(i,:);
end
ftsize = 30
nbsize = 30
% lsize  = 4;

figure;
sumplot1 = sumplot1./2;
sumplot2 = sumplot2./2;
plot(sumplot1,'r*','LineWidth',lsize);hold on
plot(sumplot2,'b*','LineWidth',lsize)
legend('simple layered model','marmousi velocity model')
set(gca, 'LineWidth', lsize);
set(gca,'FontSize',nbsize,'FontName','Arial','XAxisLocation','bottom');
set(gcf,'unit','normalized','position',[0.1,0.1,0.25,0.8]);
ylabel('Time(s)','Fontsize',ftsize,'fontweight','normal');
xlabel('Trace','Fontsize',ftsize,'fontweight','normal');
ylabel('Average SNR of 50 Test Datasets (dB)','Fontsize',ftsize,'fontweight','normal');
% yticks(0:2:14);
xlabel('50 models training with different time dithering range','Fontsize',ftsize,'fontweight','normal');
exportgraphics(gcf,'./Fig/Models_timed.pdf','ContentType','vector')

figure;
sumplot1_2 = sumplot1_2./2;
sumplot2_2 = sumplot2_2./2;

plot(sumplot1_2,'r*','LineWidth',lsize);hold on
plot(sumplot2_2,'b*','LineWidth',lsize)
legend('simple layered model','marmousi velocity model')
set(gca, 'LineWidth', lsize);
set(gca,'FontSize',nbsize,'FontName','Arial','XAxisLocation','bottom');
set(gcf,'unit','normalized','position',[0.1,0.1,0.25,0.8]);
ylabel('Time(s)','Fontsize',ftsize,'fontweight','normal');
xlabel('Trace','Fontsize',ftsize,'fontweight','normal');
ylabel('Average SNR of 50 Models (dB)','Fontsize',ftsize,'fontweight','normal');
% yticks(0:2:14);
xlabel('50 test dataset with different time dithering range ','Fontsize',ftsize,'fontweight','normal');
exportgraphics(gcf,'./Fig/Test_timed.pdf','ContentType','vector')



% clip = 0.2;
% mm = seis(2);
% figure;
% ii = 1;
% for i = 1:2:20
%     data = zread([datapath,'/dither_test/test_data1/obser_',num2str(i, '%02d'),'.dat'],[256,128]);
%     subplot(1,10,ii);ii = ii+1;
%     zfig(data,clip,mm);
%     % max_snr_marmousi(i) = max(snr);
% end

dthir = zread([datapath,'/dither_test/test_data1/dither_',num2str(20),'.dat']);
max(dthir)-min(dthir)