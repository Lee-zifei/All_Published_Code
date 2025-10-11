clear;clc;close all;delete(gcp('nocreate'))
% %Linux
% if exist('/media/lzf/Work/code/matlab/mat_toolbox/myprogs','dir')
%     addpath('/media/lzf/Work/code/matlab/mat_toolbox/myprogs');
%     addpath('/media/lzf/Work/code/matlab/mat_toolbox/CurveLab-2.1.3/fdct_wrapping_matlab');
%     addpath(genpath('/media/lzf/Work/code/matlab/mat_toolbox/crewes'));
%     datapath='/media/lzf/Work/data';
% elseif exist('L:\code\matlab\mat_toolbox\myprogs','dir')
% %Windows
%     addpath('L:\code\matlab\mat_toolbox\myprogs');
%     addpath('L:\code\matlab\mat_toolbox\CurveLab-2.1.3\fdct_wrapping_matlab');
%     addpath(genpath('L:\code\matlab\mat_toolbox\crewes'));
%     datapath='L:\data';
% elseif exist('/data/data1/lzf/code/matlab/mat_toolbox/myprogs','dir')
% %Server
%     addpath('/data/data1/lzf/code/matlab/mat_toolbox/myprogs');
%     addpath('/data/data1/lzf/code/matlab/mat_toolbox/CurveLab-2.1.3/fdct_wrapping_matlab');
%     addpath(genpath('/data/data1/lzf/code/matlab/mat_toolbox/rewes'));
%     datapath='/data/data1/lzf/data';
% else
% %MAC
%     addpath('/Users/zifeilee/code/matlab/mat_toolbox/myprogs');
%     addpath('/Users/zifeilee/code/matlab/mat_toolbox/CurveLab-2.1.3/fdct_wrapping_matlab');
%     addpath(genpath('/Users/zifeilee/code/matlab/mat_toolbox/crewes'));
% end
%###################################################################################################
addpath('myprogs');

agc= 0;      %是否使用增益后数据
n1 = 2500;
n2 = 621;
k = 10;
dt = 0.004;  %控制激发编码范围，dt越小越离散

source1 = 300;
source2 = 420;
if agc==1
    filename1 = ['./data/crg',num2str(source1),'agc.dat'];
    filename2 = ['./data/crg',num2str(source2),'agc.dat'];
    clip=5;
else
    filename1 = ['./data/crg',num2str(source1),'.dat'];
    filename2 = ['./data/crg',num2str(source2),'.dat'];
    clip=1e-2;
end

d1 = zread(filename1);
d1 = reshape(d1, [n1 n2]);
% zfig(d1)

d2 = zread(filename2);
d2 = reshape(d2,[n1 n2]);

delay_averange=zeros(1,15);
delay_variance=zeros(1,15);
snr_max=zeros(1,15);

%% generate blended data
% the coefficient of delay
randn('seed',20220902)
%*****************************产生随机延迟时间，并且控制随机数小于1*****************************
temp = randn(1,n2);
ran = temp ./ max((temp));
delay1 = abs(k*10/10*ran./(max(ran)));
delay = floor(delay1/dt);
d1_blend = d1 + dither1(d2,-delay);
d2_blend = d2 + dither1(d1,delay);
%% deblending begin
for k=1
    %*****************************d1和d2是两个单震源数据*****************************
    [n11,n22] = size(d1_blend);
    d1_new = zeros(n11, n22);
    d2_new = zeros(n11, n22);
    d1t = zeros(n11, n22);
    d2t = zeros(n11, n22);
    t = 1;
    niter=40;
    mode=1;
    snr1 = zeros(1,niter);
    error1=zeros(1,niter);
    for iter =1:niter
        d1t=d1_blend-dither1(d2_new,-delay);
        d2t=d2_blend-dither1(d1_new,delay);

        % 对信号做二维傅里叶变换
        F1 = fft2(d1t);
        F2 = fft2(d2t);

        % 计算最大幅值（用于确定阈值）
        MAX1 = max(abs(F1(:)));
        MAX2 = max(abs(F2(:)));

        % 计算动态阈值
        tau_min1 = 1e-3;
        tau_max1 = MAX1;
        tau1 = (tau_min1/ tau_max1) ^ (1.8*(iter - 1) / (niter -1)) * tau_max1;

        tau_min2 = 1e-3;
        tau_max2 = MAX2;
        tau2 = (tau_min2/ tau_max2) ^ (1.8*(iter - 1) / (niter -1)) * tau_max2;

        % 阈值处理
        if mode == 1
            % 硬阈值
            F1(abs(F1) < tau1) = 0;
            F2(abs(F2) < tau2) = 0;

        elseif mode == 2
            % 软阈值
            F1 = F1 .* max(1 - tau1 ./ abs(F1 + (F1==0)), 0);
            F2 = F2 .* max(1 - tau2 ./ abs(F2 + (F2==0)), 0);

        elseif mode == 3
            % firm阈值（这里我们仿照曲波写法）
            mask1 = abs(F1) > tau1/2;
            F1 = F1 .* mask1;
            F1 = F1 .* max(1 - tau1 ./ (abs(F1 + (F1==0))/2), 0);

            mask2 = abs(F2) > tau2/2;
            F2 = F2 .* mask2;
            F2 = F2 .* max(1 - tau2 ./ (abs(F2 + (F2==0))/2), 0);
        end
        clip = 0.01;
        mm = seis(2);
        figure(1)
        subplot()
        imagesc([d1_new],[-clip,clip]);colormap(mm);
        % 傅里叶反变换，返回时域信号
        d1_new = real(ifft2(F1));
        d2_new = real(ifft2(F2));
        %% 切除范围外数据
        k1 = 4;
        x1 = source1;
        t1 = 1;
        x2 = source2;
        t2 = 1;

        d1_new = mutter(d1_new, x1, t1, k1);
        d2_new = mutter(d2_new, x2, t2, k1);

        clc
        kk=k/10
        snr1(iter)= 10*log10(sum(sum(d1.*d1))/sum(sum((d1-d1_new).*(d1-d1_new))))


    end

end
clip = 0.01;
mm = seis(2);
imagesc([d1,d1_blend,d1_new,d1-d1_new],[-clip,clip]);colormap(mm);










