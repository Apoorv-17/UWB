
%%Readme 
%   Extract the zip folder named as UWB_gesture_CNN_Evaluation_code and all 
%   its sub folders in same directory

% The uploaded UWB-Gestures dataset contains 100 samples from each user
% each sample is saved as png image and fed as input to CNN to generate the
%% Data preapataions
[filepath,~,~] = fileparts(which('CNN_code.m'))
gesture_images = strcat(filepath,'\','Input_Training_Data')

% image data store
imds = imageDatastore(gesture_images, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
numTrainFiles = 632; %out of total 800 files
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
%%Training and Evaluation Process
layers = [imageInputLayer([ 189 90 1]);
    
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer   
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
   
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(11)
    softmaxLayer
    classificationLayer];

miniBatchsize = 20;
options = trainingOptions('adam', ...
    'InitialLearnRate',3e-4, ...
    'SquaredGradientDecayFactor',0.99, ...
    'MaxEpochs',20, ...
    'ValidationData',imdsValidation, ...   
    'ValidationFrequency',10, ...    
    'MiniBatchSize',32, ...
    'Plots','training-progress');
[NeT , tr]= trainNetwork(imdsTrain,layers,options);
YPred = classify(NeT,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)