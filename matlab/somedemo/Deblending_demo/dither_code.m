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
addpath(genpath('CurveLab-2.1.3/fdct_wrapping_matlab'));
clip = 0.01;
mm = seis(2);
agc= 0;      %是否使用增益后数据
n1 = 2500;
n2 = 621;
k = 1;
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
figure(1)
ii = 0;
for k=1
    %*****************************d1和d2是两个单震源数据*****************************
    [n11,n22] = size(d1_blend);
    d1_new = zeros(n11, n22);
    d2_new = zeros(n11, n22);
    d1t = zeros(n11, n22);
    d2t = zeros(n11, n22);
    t = 1;
    niter=30;
    mode=1;
    snr1 = zeros(1,niter);
    error1=zeros(1,niter);
    for iter =1:niter
        d1t=d1_blend-dither1(d2_new,-delay);
        d2t=d2_blend-dither1(d1_new,delay);

        C1 = fdct_wrapping(d1t,1,2);
        C2 = fdct_wrapping(d2t,1,2);
        %     % Apply thresholding

        MAX1 = 0;
        MAX2 = 0;
        MA1 = [];
        MA2 = [];
        for s = 1:length(C1)
            for w = 1:length(C1{s})
                MA1(s,w)=max(max(abs(C1{s}{w})));
                MA2(s,w)=max(max(abs(C2{s}{w})));
            end
        end
        MAX1=max(max(MA1));
        tau_max1 = MAX1 ;
        tau_min1 = 1e-3;
        tau1 = (tau_min1/ tau_max1) ^ (1.2*(iter - 1) / (niter -1)) * tau_max1;
        tau1
        MAX2=max(max(MA2));
        tau_max2 = MAX2 ;
        tau_min2 = 1e-3;
        tau2 = (tau_min2/ tau_max2) ^ (1.2*(iter - 1) / (niter -1)) * tau_max2;
        %%
        E=[];
        for s = 1:length(C1)
            E{s} = cell(size(C1{s}));
            for w = 1:length(C1{s})
                E{s}{w}=ones(size(C1{s}{w}));
                if mode==1
                    %%%hard threshold %%%%%%
                    C1{s}{w}=C1{s}{w}.*(abs(C1{s}{w})>=tau1*E{s}{w});
                    C2{s}{w}=C2{s}{w}.*(abs(C2{s}{w})>=tau1*E{s}{w});
                end
                if mode==2
                    %%%soft threshold %%%%%%
                    C1{s}{w}=C1{s}{w}.*max(1-tau1./abs(C1{s}{w})+(C1{s}{w}==0),0);
                    C2{s}{w}=C2{s}{w}.*max(1-tau2./abs(C2{s}{w})+(C2{s}{w}==0),0);
                end
                if mode==3
                    %%%firm threshold %%%%%%
                    C1{s}{w}=C1{s}{w}.*(abs(C1{s}{w})>tau1*E{s}{w}/2);
                    C2{s}{w}=C2{s}{w}.*(abs(C2{s}{w})>tau2*E{s}{w}/2);
                    C1{s}{w}=C1{s}{w}.*max(1-tau1./abs(C1{s}{w}/2)+(C1{s}{w}==0),0);
                    C2{s}{w}=C2{s}{w}.*max(1-tau2./abs(C2{s}{w}/2)+(C2{s}{w}==0),0);
                end
            end
        end
        %% 切除范围外数据
        k1 = 4;
        x1 = source1;
        t1 = 1;
        x2 = source2;
        t2 = 1;
        d1_new = real(ifdct_wrapping(C1,1,n11,n22));
        d2_new = real(ifdct_wrapping(C2,1,n11,n22));
        d1_new = mutter(d1_new, x1, t1, k1);
        d2_new = mutter(d2_new, x2, t2, k1);
        
        if mod(iter,2) == 0
            ii = ii+1;
            subplot(3,5,ii);
            imagesc([d1_new],[-clip,clip]);colormap(mm);
        end
            % clc
        % kk=k/10
        % snr1(iter)= 10*log10(sum(sum(d1.*d1))/sum(sum((d1-d1_new).*(d1-d1_new))))

        
    end

end
figure(2);
imagesc([d1,d1_blend,d1_new,d1-d1_new],[-clip,clip]);colormap(mm);


    



    



