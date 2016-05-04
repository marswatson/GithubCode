clear all;
% neural net with raw pixel
display('neural nets with raw pixels...');
Files = dir('00\*.jpg');
LengthFiles = length(Files);

N_classes = 33;
N_eachClass = 26;
LengthFiles = N_classes * N_eachClass;

inputs = [];
for i = 1:LengthFiles;
    img = imread(strcat('00\',Files(i).name));
    img = reshape(img, [32*32,1]);
    inputs = [inputs img];
end

inputs = im2double(inputs);
targets = [];
for i = 1:N_classes;
    for n = 1:N_eachClass;
        s = sparse(i,1,1,N_classes,1);
        targets = [targets, s];
    end
end
targets = full(targets);


% Create a Pattern Recognition Network
hiddenLayerSize = 80;
net = patternnet(hiddenLayerSize);
net = configure(net,inputs,targets);


% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 10/100;


% Train the Network
net = init(net);
net.trainFcn = 'trainoss';
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs)

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
 figure, plotconfusion(targets,outputs)
% figure, ploterrhist(errors)
save('trained_net_std.mat','net');