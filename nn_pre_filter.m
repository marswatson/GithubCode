
clear all;
Files = dir('C:\Users\Administrator\Desktop\Capstone\CNN_data_uni2_32\00\*.jpg');
LengthFiles = length(Files);
inputs = [];

N_fil = 10;
N_classes = 33;
N_eachClass = 26;
LengthFiles = N_classes * N_eachClass;


display('Filtering...');
r = [];
for i = 1:LengthFiles;
    img = imread(strcat('C:\Users\Administrator\Desktop\Capstone\CNN_data_uni2_32\00\',Files(i).name));
    img = im2double(img);
    F=makeRFSfilters;
    for k = 1:N_fil;
        responses(:,:,k) = conv2(img,F(:,:,k),'same');
    end
    
    r_eachImg = [];
    for x = 1:32;
        for y = 1:32;
            r_pixel = reshape(responses(x, y, :), [1, N_fil]);
            r_eachImg = [r_eachImg; r_pixel];
        end
    end
    r = [r; r_eachImg];
end


display('clustering...');
K = 150;
[idx,centroids] = kmeans(r, K);



% creating histogram
display('creating histogram...');
H = [];

for i = 1:LengthFiles;
    img = imread(strcat('C:\Users\Administrator\Desktop\Capstone\CNN_data_uni2_32\00\',Files(i).name));
    img = im2double(img);
    F=makeRFSfilters;
    for k = 1:N_fil;
        responses(:,:,k) = conv2(img,F(:,:,k),'same');
    end
    
    % histogram for every image
    h = zeros(1,K);
    for x = 1:32;
        for y = 1:32;
            r_pixel = reshape(responses(x, y, :), [1, N_fil]);
            
            min_d = 1000000;
            for k = 1:K;
                distance = norm( r_pixel - centroids(k,:) );
                if ( distance < min_d);
                    min_d = distance;
                    min_k = k;
                end
            end
            h(min_k) = h(min_k) + 1;
        end
    end
    H = [H; h];
end




% neural net
display('neural nets...');
inputs = H';
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
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 50/100;
net.divideParam.testRatio = 50/100;


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
save('trained_net_filters.mat','net');

