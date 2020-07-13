clc
clear all
allmuscles=cell(1615,8);
labels=cell(1615,8);
desired_EMG_length=450;
newdatasize=0;
onur=0;
surhan=0;
for j=1:50
s=importdata(sprintf('Subject%d.mat',j));
datasize=size(s.Data,2);
% EMG data filtering & interpolation
Wn=350/((s.EMGFreq)/2);
[z,p,k] = butter(6,Wn,'high');         %%By using the results from the previous line, filter is calculated
[sos,g] = zp2sos(z,p,k);

for i=1:datasize
    for kas=1:8   
        v = filtfilt(sos, g, s.Data(i).EMG(kas,:));
        v=zscore(v);
        v=smoothdata(v,'gaussian');
        %v = s.Data(i).EMG(kas,:);     %8'e kadar arttýkça 'v' farklý kasa geçiyor.        
        interp_lim = length(v);                    
        interp_step = interp_lim/desired_EMG_length;%v,x ve xq interpolasyon için istenen parametreler.
        x = 1:1:interp_lim;                          %desired_EMG_length & raw data kullanýlarak bulunuyor.
        xq = 1:interp_step:interp_lim;
        %onur.normal.Data(ergin).EMG.TA(trial,:) = interp1(x,v,xq,'spline');
        yenidata=interp1(x,v,xq,'spline');
        allmuscles{(newdatasize+i),kas}= yenidata;
        if strcmp(s.Data(i).Task,'StepUp     ')==1
        labels{(newdatasize+i),kas}='A';
        onur=onur+1; %%kaç tane StepUp var bakmak için : 169*8 Kas
        else
        labels{(newdatasize+i),kas}='N';
        surhan=surhan+1;
        end    
    end    
end
newdatasize=newdatasize+datasize;
end

%%I decided to work only with Tibialis Anterior muscle to
%%perform classification. I will later fuse the EMG data
%%coming from other 7 muscles.
tibialis=allmuscles(:,1);
label_tibialis=labels(:,1);

%%In order to classify StepUp Tasks from rest of the tasks
%%I labeled them by their categorical property
stepupcases = tibialis(strcmp(label_tibialis,'A')); 
stepupcaseslabel = label_tibialis(strcmp(label_tibialis,'A'));
allothercases = tibialis(strcmp(label_tibialis,'N'));
allothercaseslabel = label_tibialis(strcmp(label_tibialis,'N'));

%%I split the dataset into training and test datasets
%%by randomly picking with dividerand function.
[trainIndA,~,testIndA] = dividerand(169,0.8,0.0,0.2);
[trainIndN,~,testIndN] = dividerand(1446,0.8,0.0,0.2);

XTrainA = stepupcases(trainIndA);
YTrainA = stepupcaseslabel(trainIndA);

XTrainN = allothercases(trainIndN);
YTrainN = allothercaseslabel(trainIndN);

XTestA = stepupcases(testIndA);
YTestA = stepupcaseslabel(testIndA);

XTestN = allothercases(testIndN);
YTestN = allothercaseslabel(testIndN);


%%I equalized the lengths of the two new classes by 'padding??'
XTrain = [repmat(XTrainA(1:128),9,1); XTrainN(1:1152)];
YTrain = [repmat(YTrainA(1:128),9,1); YTrainN(1:1152)];

XTest = [repmat(XTestA(1:32),9,1); XTestN(1:288)];
YTest = [repmat(YTestA(1:32),9,1); YTestN(1:288)];

%%I define the parameters of the neural network
layers = [ ...
    sequenceInputLayer(1)
    bilstmLayer(100,'OutputMode','last') %%I decided to use bilstm layer because it works well with time series signals
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ]
%%I fine tuned the hyperparameters manually but it probably can further be
%%optimized to give better accuracy results.
options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 40, ... %%batch size'ý düþürmek iyi geliyor
    'InitialLearnRate', 0.02, ...
    'SequenceLength', 450, ...
    'GradientThreshold', 1, ...
    'plots','training-progress', ...
    'Verbose',false);

%%MATLAB forced me to that kind of thing  I didnt understand why but when
%%the label sequence is converted such below, there were not any problem.
YTrain=categorical(YTrain);
YTest=categorical(YTest);

%%training neural network
net = trainNetwork(XTrain,YTrain,layers,options);

%%performing classification with trained model
trainPred = classify(net,XTrain,'SequenceLength',1000);
LSTMAccuracy_train = sum(trainPred == YTrain)/numel(YTrain)*100;

figure(1)
confusionchart(YTrain,trainPred,'ColumnSummary','column-normalized','RowSummary','row-normalized','Title','Training: Confusion Chart for LSTM');

%%calculating test accuracy
testPred = classify(net,XTest,'SequenceLength',1000);
LSTMAccuracy_test = sum(testPred == YTest)/numel(YTest)*100;

figure(2)
confusionchart(YTest,testPred,'ColumnSummary','column-normalized','RowSummary','row-normalized','Title','Test: Confusion Chart for LSTM');
