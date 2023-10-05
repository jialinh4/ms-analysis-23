%This function finds the change of speed in a curve
%by running a 5 point window 


%take position data and sampling rate
%col = column of data, SR=sampling rate

function [deriv]=diff5P(Col,SR)
%[b,a] = butter(4,2*6/100,'low'); %low pass filter design
%Col=filtfilt(b,a,Col);
[m,n]=size(Col); %find length
%Col = smooth(Col,'rlowess');
%apply a Savitzky-Golay filter with a frame of 5 points and a
%power of 3 to smooth the signal
y = sgolayfilt(Col,1,15); 

dT=1/SR; %delta T is 1 sec/SR
    
dX=diff(y);%find difference between adjacent points of the new curve
dX(m)=dX(m-1);%insert point to tail of dX (diff returns m-1)

deriv=dX/dT; %find slope by dividing by delta T
