function [ dataout ] = dither1( datain, shift )
% Make a dithering to each trace of the input data
%   Detailed explanation goes here
%   datain : input data
%   shift  : random time shift for each trace
%   dataout: output data

   [nt,nx]=size(datain);
   dataout=zeros(nt,nx);

   if(size(datain,2)~=size(shift))
       error('datain and shift should have same size in space');
   end
   
   for ix=1:nx
       for it=1:nt
            itt=it+shift(ix);
            if(itt<=0 || itt>nt) continue; end
            dataout(itt,ix)=datain(it,ix);
       end
   end

end

