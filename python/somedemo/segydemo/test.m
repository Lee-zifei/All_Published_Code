% clear;clc;close all
%Linux
if exist('/media/lzf/Data/code/matlab/mat_toolbox/myprogs','dir')
    addpath('/media/lzf/Data/code/matlab/mat_toolbox/myprogs');
    addpath('/media/lzf/Data/code/matlab/mat_toolbox/CurveLab-2.1.3/fdct_wrapping_matlab');
    addpath(genpath('/media/lzf/Data/code/matlab/mat_toolbox/crewes'));
    datapath='/media/lzf/Data/data';
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
mm = seis(2);
addpath(genpath('myprogs'));
if exist('Data')
    close all;
else
    clear;clc;close all;
    [Data,SegyTraceHeaders,SegyHeader]=ReadSegy('./data/item2_mig.sgy');
end
n1 = 1001;
n2 = 401;
n3 = 101;
domain = 'Inline';                                     % select domain

for k=1800;                                            % ganther number                                 
    disp(k);
    p1=find(k==[SegyTraceHeaders.cdp]);
    p2=find(k==[SegyTraceHeaders.Inline3D]);           % This post stack 3Ddata have 2 key words:CDP and Inline3D
    if strcmp(domain, 'CDP')
        p = p1;
        patch = zeros(n1,n3);
    elseif strcmp(domain, 'Inline')
        p = p2;
        patch = zeros(n1,n2);
    end
    for i=1:length(p)
        patch(:,i) = Data(:,p(i));                    % if you want see one domain's patch,you need to select the trace
    end

end
filename = './data/data.dat';
fid = fopen(filename,'w');                          % save .dat type data
fwrite(fid,Data,'float');
fclose(fid);

figure(1);
clip = 10000 ;
% mm =seis(2);
% zfig(patch,clip,mm);
imagesc(patch,[-clip,clip]);colormap(mm);


                                                      % read .dat data and fig
g=1;                                                  % select domain
k=101;                                                % g=1 means the first domain,CDP,k=101 mains CDP key words from 1700 to 1800 is the 101th trace
if g == 1
    signal = zeros(n1,n2);
    fid  = fopen(filename, "r");
    fseek(fid, n1*n2*(k-1)*4,'bof');
    signal =fread (fid,[n1 n2], "float");
    fclose(fid);
elseif g == 2
    signal = zeros(n1,n3);
    fid  = fopen(filename, "r");
    for j=1:n3
        fseek(fid, n1*n2*(j-1)*4+(k-1)*n1*4,'bof');
        signal(:,j) =fread (fid,n1, "float");
    end
    fclose(fid);
end
figure(2);
imagesc(signal,[-clip,clip]);colormap(mm);
figure(3);
imagesc(signal-patch,[-clip,clip]);colormap(mm);  %Two ways to read segy data ,the result is same

%if the data size is smoll,you can read 3d data and reshapr[n1,n2,n3] to
%get the all data ,but this way need your computer have a large Ram,the
%same with python.:wq
