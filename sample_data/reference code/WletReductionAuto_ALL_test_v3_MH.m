% Batch processing of subjects: RETEST

%path=(path,'/Users/manueleh/Desktop/PD_Brasic_sample/Analysis/')

% subject_ID 
NameF1='PDMotion';
NameF2=['0001';'0002';'0003';'0006';'0007';'0008';'0009';'0010';
    '0011';'0012';'0013';'0015';'0017';'0018';'0019';'0020';
    '0021';'0022';'0023';'0024';'0025';'0026';'0027';'0028';'0029';'0030';'0005';'0014'];
NameF3='.xlsx';
NameF4='Test';
Name4='.jpg';

% Components
NameA2=["_PS_";"_HM_";"_FT_";"_TT_";"_LA_"];
NameU3=["rfx";"rfy";"rfz";"rwx";"rwy";"rwz";"lfx";"lfy";"lfz";"lwx";"lwy";"lwz"];
NameL3=["rax";"ray";"raz";"rtx";"rty";"rtz";"lax";"lay";"laz";"ltx";"lty";"ltz"];
NameTabs = ["pro-sup";"hand motion";"finger tap";"toe tap";"leg agility"];

s = '/';
u = '_';
home_path = '/Users/manueleh/Desktop/PD_Brasic_sample/';
SR=80;%Hz  

% initialize start/stop matrix
st_1 = zeros(28,5,4); % original start
st_2 = zeros(28,5,4);
st_3 = zeros(28,5,4); % revised start
st_4 = zeros(28,5,4);


% loop through every subject
for subj = [1:26]%(size(NameF2,1)-2)] % fix 6 later (PD), fix 6 later (Control), fix 1 (Control, RT), fix 8 (PD, RT)
    %% importing data

    Acc=importdata(strcat(home_path,'All_combine_sheets',s,NameF2(subj,:),s,NameF1,u,NameF2(subj,:),u,NameF4,NameF3));
    %CompPtr={Acc.data.PS;Acc.data.HM;Acc.data.FT};

    CompPtr={Acc.data.PS;Acc.data.HM;Acc.data.FT;Acc.data.TT;Acc.data.LA};

    for i = 1:5 % 1-3 for 5,14, 5 for all else
        IntPtr=CompPtr{i}; %
        %L = length(IntPtr);

        Name2 = NameA2(i);
        for j = 4:3:13 % for specifica sensor
            k = (j-1)/3;
            
            % initialize temporary variables
            x1=zeros(1,10000);
            var=zeros(1,10000); % envolope

            x1=IntPtr(:,j); x2=IntPtr(:,j+1); x3=IntPtr(:,j+2); % x,y,x
            v1=diff5P(x1,SR); v2=diff5P(x1,SR); v3=diff5P(x1,SR); % jerk

            v_all = sqrt(v1.^2 + v2.^2 + v3.^2); % magnitude of jerk

            % smooth signal using RMS
            maxvar=0.0; maxindex=1;
            for c3=1:size(v_all)-100 %100 sample data window 1.2 second 
                avg=sum(v_all(c3:c3+100))/101;
                rms=0.0;
                for c4=c3:c3+100
                    rms= rms + (v_all(c4)-avg)*(v_all(c4)-avg);
                end
                var(c3)=rms/101;
                if(var(c3)>maxvar)
                    maxvar=var(c3);
                    maxindex=c3;
                end
            end

            % find initial and final crossing points - using max as reference   
            varavg = maxvar/6; % 1/6 of the max rms

            up = find(var(1:maxindex)<varavg,1,'last');
            down = find(var(maxindex:end)<varavg,1,'first');
            down = maxindex+down-1;

            % if up is empty, use first sample
            if isempty(up)
                up=1;
            end
            
            st_1(subj,i,k) = up;
            st_2(subj,i,k) = down;
            
            figure((subj*100)+(i*10)+(k))
            plot(var); hold on; plot(up,var(up),'og',down,var(down),'or'); 
            
        end
    end
end
   
% last 2 subjects
for subj = [27:size(NameF2,1)] % fix 6 later (PD), fix 6 later (Control), fix 1 (Control, RT), fix 8 (PD, RT)
    %% importing data

    Acc=importdata(strcat(home_path,'All_combine_sheets',s,NameF2(subj,:),s,NameF1,u,NameF2(subj,:),u,NameF4,NameF3));
    CompPtr={Acc.data.PS;Acc.data.HM;Acc.data.FT};

    %CompPtr={Acc.data.PS;Acc.data.HM;Acc.data.FT;Acc.data.TT;Acc.data.LA};

    for i = 1:3 % 1-3 for 5,14, 5 for all else
        IntPtr=CompPtr{i};
        %L = length(IntPtr);

        Name2 = NameA2(i);
        for j = 4:3:13
            k = (j-1)/3;
            
            % initialize temporary variables
            x1=zeros(1,10000);
            var=zeros(1,10000);
            x1=IntPtr(:,j); x2=IntPtr(:,j+1); x3=IntPtr(:,j+2);
            v1=diff5P(x1,SR); v2=diff5P(x1,SR); v3=diff5P(x1,SR);% jerk

            v_all = sqrt(v1.^2 + v2.^2 + v3.^2); % magnitude of jerk

            % smooth signal using RMS
            maxvar=0.0; maxindex=1;
            for c3=1:size(v_all)-100
                avg=sum(v_all(c3:c3+100))/101;
                rms=0.0;
                for c4=c3:c3+100
                    rms= rms + (v_all(c4)-avg)*(v_all(c4)-avg);
                end
                var(c3)=rms/101;
                if(var(c3)>maxvar)
                    maxvar=var(c3);
                    maxindex=c3;
                end
            end

            % find initial and final crossing points - using max as reference   
            varavg = maxvar/6;

            up = find(var(1:maxindex)<varavg,1,'last');
            down = find(var(maxindex:end)<varavg,1,'first');
            down = maxindex+down-1;

            % if up is empty, use first sample
            if isempty(up)
                up=1;
            end
            
            st_1(subj,i,k) = up;
            st_2(subj,i,k) = down;
            
            figure((subj*100)+(i*10)+(k))
            plot(var); hold on; plot(up,var(up),'og',down,var(down),'or'); 
            
        end
    end
end


st_flag = zeros(28,5,4); % flag trials where start significantly differ

% create corrected start/stop values
for i=1:28
    for j=1:5
        b(1:4) = (st_2(i,j,:)-st_1(i,j,:)); % duration
        sb(1:4) = st_1(i,j,:); %start
        
        sb1 = round(mean(sb(1:2))) + round(mean(b(1:2))/2) - 500; % right start
        sb2 = round(mean(sb(3:4))) + round(mean(b(3:4))/2) - 500; % left start
        
        % error-check
        if sb1 < 1;
            sb1 = 1;
        end
        
        if sb2 < 1;
            sb2 = 1;
        end
        
        st_flag(i,j,:) = [[sb(1:2)-round(mean(sb(1:2)))],[sb(3:4)-round(mean(sb(3:4)))]];
        st_3(i,j,:) = [sb1 sb1 sb2 sb2];
        st_4(i,j,:) = [sb1+800 sb1+800 sb2+800 sb2+800];
        
    end
end

% save start/stop values for Retest data
save('startstop_test.mat','st_1','st_2','st_3','st_4','st_flag')
        
for subj = [1:(size(NameF2,1)-2)] % fix 6 later (PD), fix 6 later (Control), fix 1 (Control, RT), fix 8 (PD, RT)
    %% importing data

    Acc=importdata(strcat(home_path,'All_combine_sheets',s,NameF2(subj,:),s,NameF1,u,NameF2(subj,:),u,NameF4,NameF3));
    %CompPtr={Acc.data.PS;Acc.data.HM;Acc.data.FT};

    CompPtr={Acc.data.PS;Acc.data.HM;Acc.data.FT;Acc.data.TT;Acc.data.LA};

    for i = 1:5 % 1-3 for 5,14, 5 for all else
        IntPtr=CompPtr{i};
        %L = length(IntPtr);

        Name2 = NameA2(i);
        for j = 4:3:13
            k = (j-1)/3; 
            
            start = st_3(subj,i,k); %st_1{subj}{i}(k);
            stop = st_4(subj,i,k); %st_2{subj}{i}(k);
            
            %rat = num2str(rt{subj}{i}(k));
            %time = IntPtr(start:stop,1);
            x1 = sqrt(IntPtr(start:stop,j).^2+IntPtr(start:stop,j+1).^2+IntPtr(start:stop,j+2).^2);
            %[A.(strcat(NameF2,Name2,Name3,Name4))] = x1; find way to add stuff to Excel sheet!   
            Name3=NameU3(j-3);

            cwt(x1,'amor',80)

            if i == 1
                if (j == 4) | (j == 10)
                    title({'Pronation-Supination'; 'Finger'},'FontSize', 12,'FontWeight', 'bold')
                elseif (j == 7) | (j == 13)
                    title({'Pronation-Supination'; 'Wrist'},'FontSize', 12,'FontWeight', 'bold')
                end
            elseif i == 2
                if (j == 4) | (j == 10)
                    title({'Hand Movement'; 'Finger'},'FontSize', 12,'FontWeight', 'bold')
                elseif (j == 7) | (j == 13)
                    title({'Hand Movement'; 'Wrist'},'FontSize', 12,'FontWeight', 'bold')
                end
            elseif i == 3
                if (j == 4) | (j == 10)
                    title({'Finger Tapping'; 'Finger'},'FontSize', 12,'FontWeight', 'bold')
                elseif (j == 7) | (j == 13)
                    title({'Finger Tapping'; 'Wrist'},'FontSize', 12,'FontWeight', 'bold')
                end
            elseif i == 4
                if (j == 4) | (j == 10)
                    title({'Toe Tapping'; 'Ankle'},'FontSize', 12,'FontWeight', 'bold')
                elseif (j == 7) | (j == 13)
                    title({'Toe Tapping'; 'Toe'},'FontSize', 12,'FontWeight', 'bold')
                end
            elseif i == 5    
                if (j == 4) | (j == 10)
                    title({'Leg Agility'; 'Ankle'},'FontSize', 12,'FontWeight', 'bold')
                elseif (j == 7) | (j == 13)
                    title({'Leg Agility'; 'Toe'},'FontSize', 12,'FontWeight', 'bold')
                end
            end    

            print(gcf,strcat('Wlet',NameF2(subj,:),Name2,Name3,NameF4,Name4),'-djpeg','-r300')



            %saveas(gcf,strcat(Name1,Name2,Name3,Name5(2),'TB',Name4));

    %         [cfs,frq] = cwt(x1,'amor',80); % 80 Hz sampling rate
    %         tms = (0:numel(x1)-1)/80;
    %         
    %         figure(2)
    %         subplot(2,1,1)
    %         plot(tms,x1)
    %         axis tight
    %         title('Signal and Scalogram')
    %         xlabel('Time (s)')
    %         ylabel('Amplitude')
    %         subplot(2,1,2)
    %         surface(tms,frq,abs(cfs))
    %         axis tight
    %         shading flat
    %         xlabel('Time (s)')
    %         ylabel('Frequency (Hz)')
    %         set(gca,'yscale','log')


            %im = ind2rgb(im2uint8(rescale(abs(cfs))),jet(128));

            %figure; 
            %cwt(x1,'bump'); %creating and saving figure
            %cfs = abs(cwt(x1));
            %im = ind2rgb(im2uint8(rescale(cfs)),jet(128));
            %imwrite(imresize(im,[224 224]),strcat(home_path,'figures',s,'Wlet',NameF2(subj,:),u,Name2,u,rat,u,Name3,Name4));

        end
        %Write out combined data
        %writematrix(A,strcat(NameF5,NameF2,NameF3),'Sheet',NameTabs(i));
    end
end   
    
        
for subj = [27:size(NameF2,1)] % fix 6 later (PD), fix 6 later (Control), fix 1 (Control, RT), fix 8 (PD, RT)
    %% importing data

    Acc=importdata(strcat(home_path,'All_combine_sheets',s,NameF2(subj,:),s,NameF1,u,NameF2(subj,:),u,NameF4,NameF3));
    CompPtr={Acc.data.PS;Acc.data.HM;Acc.data.FT};

    %CompPtr={Acc.data.PS;Acc.data.HM;Acc.data.FT;Acc.data.TT;Acc.data.LA};

    for i = 1:3 % 1-3 for 5,14, 5 for all else
        IntPtr=CompPtr{i};
        %L = length(IntPtr);

        Name2 = NameA2(i);
        for j = 4:3:13
            k = (j-1)/3; 
            
            start = st_3(subj,i,k); %st_1{subj}{i}(k);
            stop = st_4(subj,i,k); %st_2{subj}{i}(k);
            
            %rat = num2str(rt{subj}{i}(k));
            %time = IntPtr(start:stop,1);
            x1 = sqrt(IntPtr(start:stop,j).^2+IntPtr(start:stop,j+1).^2+IntPtr(start:stop,j+2).^2);
            %[A.(strcat(NameF2,Name2,Name3,Name4))] = x1; find way to add stuff to Excel sheet!   
            Name3=NameU3(j-3);

            cwt(x1,'amor',80)

            if i == 1
                if (j == 4) | (j == 10)
                    title({'Pronation-Supination'; 'Finger'},'FontSize', 12,'FontWeight', 'bold')
                elseif (j == 7) | (j == 13)
                    title({'Pronation-Supination'; 'Wrist'},'FontSize', 12,'FontWeight', 'bold')
                end
            elseif i == 2
                if (j == 4) | (j == 10)
                    title({'Hand Movement'; 'Finger'},'FontSize', 12,'FontWeight', 'bold')
                elseif (j == 7) | (j == 13)
                    title({'Hand Movement'; 'Wrist'},'FontSize', 12,'FontWeight', 'bold')
                end
            elseif i == 3
                if (j == 4) | (j == 10)
                    title({'Finger Tapping'; 'Finger'},'FontSize', 12,'FontWeight', 'bold')
                elseif (j == 7) | (j == 13)
                    title({'Finger Tapping'; 'Wrist'},'FontSize', 12,'FontWeight', 'bold')
                end
            elseif i == 4
                if (j == 4) | (j == 10)
                    title({'Toe Tapping'; 'Ankle'},'FontSize', 12,'FontWeight', 'bold')
                elseif (j == 7) | (j == 13)
                    title({'Toe Tapping'; 'Toe'},'FontSize', 12,'FontWeight', 'bold')
                end
            elseif i == 5    
                if (j == 4) | (j == 10)
                    title({'Leg Agility'; 'Ankle'},'FontSize', 12,'FontWeight', 'bold')
                elseif (j == 7) | (j == 13)
                    title({'Leg Agility'; 'Toe'},'FontSize', 12,'FontWeight', 'bold')
                end
            end    

            print(gcf,strcat('Wlet',NameF2(subj,:),Name2,Name3,NameF4,Name4),'-djpeg','-r300')



            %saveas(gcf,strcat(Name1,Name2,Name3,Name5(2),'TB',Name4));

    %         [cfs,frq] = cwt(x1,'amor',80); % 80 Hz sampling rate
    %         tms = (0:numel(x1)-1)/80;
    %         
    %         figure(2)
    %         subplot(2,1,1)
    %         plot(tms,x1)
    %         axis tight
    %         title('Signal and Scalogram')
    %         xlabel('Time (s)')
    %         ylabel('Amplitude')
    %         subplot(2,1,2)
    %         surface(tms,frq,abs(cfs))
    %         axis tight
    %         shading flat
    %         xlabel('Time (s)')
    %         ylabel('Frequency (Hz)')
    %         set(gca,'yscale','log')


            %im = ind2rgb(im2uint8(rescale(abs(cfs))),jet(128));

            %figure; 
            %cwt(x1,'bump'); %creating and saving figure
            %cfs = abs(cwt(x1));
            %im = ind2rgb(im2uint8(rescale(cfs)),jet(128));
            %imwrite(imresize(im,[224 224]),strcat(home_path,'figures',s,'Wlet',NameF2(subj,:),u,Name2,u,rat,u,Name3,Name4));

        end
        %Write out combined data
        %writematrix(A,strcat(NameF5,NameF2,NameF3),'Sheet',NameTabs(i));
    end
end   
       
%% Stitch together images

% Defaults for this blog post
width = 5;     % Width in inches
height = 9;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize



% Set the default Size for display
defpos = get(0,'defaultFigurePosition');
set(0,'defaultFigurePosition', [defpos(1) defpos(2) width*100, height*100]);

% Set the defaults for saving/printing to a file
set(0,'defaultFigureInvertHardcopy','on'); % This is the default anyway
set(0,'defaultFigurePaperUnits','inches'); % This is the default anyway
defsize = get(gcf, 'PaperSize');
left = (defsize(1)- width)/2;
bottom = (defsize(2)- height)/2;
defsize = [left, bottom, width, height];
set(0, 'defaultFigurePaperPosition', defsize);




for subj = [1:26] % fix 6 later (PD), fix 6 later (Control), fix 1 (Control, RT), fix 8 (PD, RT)
    Name1 = strcat('Wlet',NameF2(subj,:),'Test');
    
    %% importing data
    for i = 1:3 % loop through UE task
        Name2 = NameA2(i);
        Name3=NameU3(1);
        temp = strcat(home_path,'Analysis',s,'Full_Test',s,'Wlet',NameF2(subj,:),Name2,Name3,NameF4,Name4);
        A = imread(temp); 
        
        Name3=NameU3(4);
        temp = strcat(home_path,'Analysis',s,'Full_Test',s,'Wlet',NameF2(subj,:),Name2,Name3,NameF4,Name4);
        B = imread(temp); 
        
        
        subplot(2,1,1)
        image(A)
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        ylabel('Finger','FontSize', 12)
        
        if i == 1
                title('Pronation-Supination','FontSize', 12,'FontWeight', 'bold')
        elseif i == 2
                title('Hand Movement','FontSize', 12,'FontWeight', 'bold')
        elseif i == 3
                title('Finger Tapping','FontSize', 12,'FontWeight', 'bold')        
        elseif i == 4
                title('Toe Tapping','FontSize', 12,'FontWeight', 'bold')
        elseif i == 5
                title('Leg Agility','FontSize', 12,'FontWeight', 'bold')
        end
        
        subplot(2,1,2)
        image(B)
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        ylabel('Wrist','FontSize', 12)
        
        print(gcf,strcat(Name1,Name2,'right',Name4),'-djpeg','-r300')
        
        Name3=NameU3(7);
        temp = strcat(home_path,'Analysis',s,'Full_Test',s,'Wlet',NameF2(subj,:),Name2,Name3,NameF4,Name4);
        A = imread(temp); 
        
        Name3=NameU3(10);
        temp = strcat(home_path,'Analysis',s,'Full_Test',s,'Wlet',NameF2(subj,:),Name2,Name3,NameF4,Name4);
        B = imread(temp); 
        
        
        subplot(2,1,1)
        image(A)
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        ylabel('Finger','FontSize', 12)
        
        if i == 1
                title('Pronation-Supination','FontSize', 12,'FontWeight', 'bold')
        elseif i == 2
                title('Hand Movement','FontSize', 12,'FontWeight', 'bold')
        elseif i == 3
                title('Finger Tapping','FontSize', 12,'FontWeight', 'bold')        
        elseif i == 4
                title('Toe Tapping','FontSize', 12,'FontWeight', 'bold')
        elseif i == 5
                title('Leg Agility','FontSize', 12,'FontWeight', 'bold')
        end
        
        subplot(2,1,2)
        image(B)
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        ylabel('Wrist','FontSize', 12)
        
        print(gcf,strcat(Name1,Name2,'left',Name4),'-djpeg','-r300')
   
    end
        
    for i = 4:5 % loop through LE task
        Name2 = NameA2(i);
        Name3=NameU3(1);
        temp = strcat(home_path,'Analysis',s,'Full_Test',s,'Wlet',NameF2(subj,:),Name2,Name3,NameF4,Name4);
        A = imread(temp); 
        
        Name3=NameU3(4);
        temp = strcat(home_path,'Analysis',s,'Full_Test',s,'Wlet',NameF2(subj,:),Name2,Name3,NameF4,Name4);
        B = imread(temp); 
        
        
        subplot(2,1,1)
        image(A)
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        ylabel('Ankle','FontSize', 12)
        
        if i == 1
                title('Pronation-Supination','FontSize', 12,'FontWeight', 'bold')
        elseif i == 2
                title('Hand Movement','FontSize', 12,'FontWeight', 'bold')
        elseif i == 3
                title('Finger Tapping','FontSize', 12,'FontWeight', 'bold')        
        elseif i == 4
                title('Toe Tapping','FontSize', 12,'FontWeight', 'bold')
        elseif i == 5
                title('Leg Agility','FontSize', 12,'FontWeight', 'bold')
        end
        
        subplot(2,1,2)
        image(B)
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        ylabel('Toe','FontSize', 12)
        
        print(gcf,strcat(Name1,Name2,'right',Name4),'-djpeg','-r300')
        
        Name3=NameU3(7);
        temp = strcat(home_path,'Analysis',s,'Full_Test',s,'Wlet',NameF2(subj,:),Name2,Name3,NameF4,Name4);
        A = imread(temp); 
        
        Name3=NameU3(10);
        temp = strcat(home_path,'Analysis',s,'Full_Test',s,'Wlet',NameF2(subj,:),Name2,Name3,NameF4,Name4);
        B = imread(temp); 
        
        
        subplot(2,1,1)
        image(A)
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        ylabel('Ankle','FontSize', 12)
        
        if i == 1
                title('Pronation-Supination','FontSize', 12,'FontWeight', 'bold')
        elseif i == 2
                title('Hand Movement','FontSize', 12,'FontWeight', 'bold')
        elseif i == 3
                title('Finger Tapping','FontSize', 12,'FontWeight', 'bold')        
        elseif i == 4
                title('Toe Tapping','FontSize', 12,'FontWeight', 'bold')
        elseif i == 5
                title('Leg Agility','FontSize', 12,'FontWeight', 'bold')
        end
        
        subplot(2,1,2)
        image(B)
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        ylabel('Toe','FontSize', 12)
        
        print(gcf,strcat(Name1,Name2,'left',Name4),'-djpeg','-r300')
   
    end            
end

    

for subj = [27:28] % fix 6 later (PD), fix 6 later (Control), fix 1 (Control, RT), fix 8 (PD, RT)
    Name1 = strcat('Wlet',NameF2(subj,:),'Test');
    
    %% importing data
    for i = 1:3 % loop through UE task
        Name2 = NameA2(i);
        Name3=NameU3(1);
        temp = strcat(home_path,'Analysis',s,'Full_Test',s,'Wlet',NameF2(subj,:),Name2,Name3,NameF4,Name4);
        A = imread(temp); 
        
        Name3=NameU3(4);
        temp = strcat(home_path,'Analysis',s,'Full_Test',s,'Wlet',NameF2(subj,:),Name2,Name3,NameF4,Name4);
        B = imread(temp); 
        
        
        subplot(2,1,1)
        image(A)
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        ylabel('Finger','FontSize', 12)
        
        if i == 1
                title('Pronation-Supination','FontSize', 12,'FontWeight', 'bold')
        elseif i == 2
                title('Hand Movement','FontSize', 12,'FontWeight', 'bold')
        elseif i == 3
                title('Finger Tapping','FontSize', 12,'FontWeight', 'bold')        
        elseif i == 4
                title('Toe Tapping','FontSize', 12,'FontWeight', 'bold')
        elseif i == 5
                title('Leg Agility','FontSize', 12,'FontWeight', 'bold')
        end
        
        subplot(2,1,2)
        image(B)
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        ylabel('Wrist','FontSize', 12)
        
        print(gcf,strcat(Name1,Name2,'right',Name4),'-djpeg','-r300')
        
        Name3=NameU3(7);
        temp = strcat(home_path,'Analysis',s,'Full_Test',s,'Wlet',NameF2(subj,:),Name2,Name3,NameF4,Name4);
        A = imread(temp); 
        
        Name3=NameU3(10);
        temp = strcat(home_path,'Analysis',s,'Full_Test',s,'Wlet',NameF2(subj,:),Name2,Name3,NameF4,Name4);
        B = imread(temp); 
        
        
        subplot(2,1,1)
        image(A)
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        ylabel('Finger','FontSize', 12)
        
        if i == 1
                title('Pronation-Supination','FontSize', 12,'FontWeight', 'bold')
        elseif i == 2
                title('Hand Movement','FontSize', 12,'FontWeight', 'bold')
        elseif i == 3
                title('Finger Tapping','FontSize', 12,'FontWeight', 'bold')        
        elseif i == 4
                title('Toe Tapping','FontSize', 12,'FontWeight', 'bold')
        elseif i == 5
                title('Leg Agility','FontSize', 12,'FontWeight', 'bold')
        end
        
        subplot(2,1,2)
        image(B)
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        ylabel('Wrist','FontSize', 12)
        
        print(gcf,strcat(Name1,Name2,'left',Name4),'-djpeg','-r300')
   
    end         
end

    
    
