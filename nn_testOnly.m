clear all

path='test_twocar\\Img8.bmp';

img=imread(path);
figure(10)
imshow(uint8(img));
carplate(path)

%load the pre-trained network
load('trained_net_std.mat'); % net is in the directory of 
% load('trained_net_filters.mat');
% load('trained_net_gaussian.mat');


Files = dir('matlabinput\*.bmp');
LengthFiles = length(Files);

numOfCharaters = 6; % 6 characters per plate
numOfPlates = floor(LengthFiles/numOfCharaters); % number of plates, can be 1 or 2

inputs = [];
for n = 1:numOfPlates
    figure(n);
    for i = 1:numOfCharaters;
        img = imread(strcat('matlabinput\',Files((n-1)*numOfCharaters+i).name));
        subplot(1,6,i);
        imshow(img);
        input = reshape(img, [32*32,1]);
        inputs = [inputs input];
    end
end
inputs = im2double(inputs);

% Test the Network
outputs = net(inputs);

% errors = gsubtract(targets,outputs);
% performance = perform(net,targets,outputs)
for i = 1:numOfPlates*numOfCharaters
    ch(i) = find(outputs(:, i) == max(outputs(:, i)));
    
    switch ch(i)
        case 1
            disp('0');
            pred(i) = {'0'};
        case 2
            disp('1');
            pred(i) = {'1'};
        case 3
            disp('2');
            pred(i) = {'2'};
        case 4
            disp('3');
            pred(i) = {'3'};
        case 5
            disp('4');
            pred(i) = {'4'};
        case 6
            disp('5');
            pred(i) = {'5'};
        case 7
            disp('6');
            pred(i) = {'6'};
        case 8
            disp('7');
            pred(i) = {'7'};
        case 9
            disp('8');
            pred(i) = {'8'};
        case 10
            disp('9');
            pred(i) = {'9'};
        case 11
            disp('A');
            pred(i) = {'A'};
        case 12
            disp('B');
            pred(i) = {'B'};
        case 13
            disp('C');
            pred(i) = {'C'};
        case 14
            disp('D');
            pred(i) = {'D'};
        case 15
            disp('E');
            pred(i) = {'E'};
        case 16
            disp('F');
            pred(i) = {'F'};
        case 17
            disp('G');
            pred(i) = {'G'};
        case 18
            disp('H');
            pred(i) = {'H'};
        case 19
            disp('J');
            pred(i) = {'J'};
        case 20
            disp('K');
            pred(i) = {'K'};
        case 21
            disp('L');
            pred(i) = {'L'};
        case 22
            disp('M');
            pred(i) = {'M'};
        case 23
            disp('N');
            pred(i) = {'N'};
        case 24
            disp('P');
            pred(i) = {'P'};
        case 25
            disp('R');
            pred(i) = {'R'};
        case 26
            disp('S');
            pred(i) = {'S'};
        case 27
            disp('T');
            pred(i) = {'T'};
        case 28
            disp('U');
            pred(i) = {'U'};
        case 29
            disp('V');
            pred(i) = {'V'};
        case 30
            disp('W');
            pred(i) = {'W'};
        case 31
            disp('X');
            pred(i) = {'X'};
        case 32
            disp('Y');
            pred(i) = {'Y'};
        case 33
            disp('Z');
            pred(i) = {'Z'};
        
    otherwise
        disp('?');
        pred(i) = '?';
    end
    figure(1)
    %text(ch(i), 40, ch(i), 'fontsize', 10, 'color', 'b') ;
end

for n = 1:floor(numOfPlates)
    figure(n);
    text(-180,40,'Predicted License: ');
    for i = 1:numOfCharaters;
        text(10*i-120, 40, pred((n-1)*numOfCharaters + i), 'fontsize', 10, 'color', 'b') ;
    end
end

% View the Network
% view(net)

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotconfusion(targets,outputs)
% figure, ploterrhist(errors)

