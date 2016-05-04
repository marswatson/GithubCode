clear all;
% neural net with histogram feature or gaussian blur
display('neural nets with histogram feature or gaussian blur...');
Files = dir('C:\Users\Administrator\Desktop\Capstone\CNN_data_uni2_32\00\*.jpg');
LengthFiles = length(Files);

N_classes = 33;
N_eachClass = 26;
LengthFiles = N_classes * N_eachClass;

inputs = [];
for i = 1:LengthFiles;
    img = imread(strcat('C:\Users\Administrator\Desktop\Capstone\CNN_data_uni2_32\00\',Files(i).name));
%     % threshold
%     threshold = 150;
%     img = img > threshold;
    % Gaussian Blur
    w = fspecial('gaussian',[5 5],3);
    img = imfilter(img,w);
%     threshold = 150;
%     bwimg = img > threshold;
%     img = reshape(bwimg, [1600,1]);
%     feature = [feature;img];
    img = reshape(img, [32*32,1]);
    inputs = [inputs img];

%     %projection histogram
%     threshold = 150;
%     bwimg = img > threshold;
%     hw = size(bwimg);
%     h_histogram = [];
%     v_histogram = [];
%     for j = 1:hw(1)
%         sum = find(bwimg(j,:)~=0);
%         h_histogram = [h_histogram,length(sum)];
%     end
%     for j = 1:hw(2)
%         sum = find(bwimg(:,j)~=0);
%         v_histogram = [v_histogram,length(sum)];
%     end
%     feature = [h_histogram,v_histogram];
%     feature = feature';
% 
%     inputs = [inputs feature];
end
% img = reshape(img, [40,40]);
% imshow(img);

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
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;


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
save('trained_net_gaussian.mat','net');