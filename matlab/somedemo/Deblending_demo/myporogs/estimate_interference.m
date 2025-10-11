function [interference] = estimate_interference(input,maintime,assistime,dt)
%   estimate_interference   : estimate the interference according to the shot
%   time                    : 2022.10.14
%   Author: Zifei Li
%   IN        input         :          A continuous aliasing signal of a seismic source
%             maintime      :          Enter the excitation time series of the source
%             assistime     :          A continuous aliasing signal of the target source
%             dt            :          Sampling time
%   OUT       interference  :          The Pseudo-deblending records with maintime

[n1, n2] = size(input);
interference = zeros(n1,n2);


for i = 1:n2
    %         temp = floor((assistime(i)-maintime)/dt);
    temp = (assistime(i)-maintime)/dt;
    [bef] =find(temp>=0 & temp<n1);
    [aft] =find(temp<=0 & temp>-n1);
    if isempty(bef) ==0
        for j=1:length(bef)
            interference(1:floor((maintime(bef(j))+n1*dt-assistime(i))/dt)+1,i) =...
                interference(1:floor((maintime(bef(j))+n1*dt-assistime(i))/dt)+1,i) +...
                input(n1-floor((maintime(bef(j))+n1*dt-assistime(i))/dt):n1,bef(j));
        end
    end
    if isempty(aft) ==0
        for j=1:length(aft)
            interference(n1-floor((assistime(i)+n1*dt-maintime(aft(j)))/dt)+1:n1,i)=...
                interference(n1-floor((assistime(i)+n1*dt-maintime(aft(j)))/dt)+1:n1,i)+...
                input(1:floor((assistime(i)+n1*dt-maintime(aft(j)))/dt),aft(j));
        end
    end

end
end


