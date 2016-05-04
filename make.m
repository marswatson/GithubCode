%// This make.m is for MATLAB  
%// Function: compile c++ files which rely on OpenCV for Matlab using mex  
  
%% Please modify your path of OpenCV  
%% If your have any question, please contact Zou Xiaoyi  
  
% Notice: first use "mex -setup" to choose your c/c++ compiler  
clear all;  
  
%-------------------------------------------------------------------  
%% get the architecture of this computer  
is_64bit = strcmp(computer,'MACI64') || strcmp(computer,'GLNXA64') || strcmp(computer,'PCWIN64');  
  
  
%-------------------------------------------------------------------  
%% the configuration of compiler  
% You need to modify this configuration according to your own path of OpenCV  
% Notice: if your system is 64bit, your OpenCV must be 64bit!  
out_dir='./';  
CPPFLAGS = ' -O -DNDEBUG -I.\ -IF:\openCV249\opencv\build\include -IF:\openCV249\opencv\build\includeopencv -IF:\openCV249\opencv\build\include\opencv2'; % your OpenCV "include" path  
LDFLAGS = ' -LF:\openCV249\opencv\build\x64\vc12\lib';                       % your OpenCV "lib" path  
LIBS = ' -lopencv_calib3d249 -lopencv_contrib249 -lopencv_core249 -lopencv_features2d249 -lopencv_flann249 -lopencv_gpu249 -lopencv_highgui249 -lopencv_imgproc249 -lopencv_legacy249 -lopencv_ml249 -lopencv_nonfree249 -lopencv_objdetect249 -lopencv_photo249 -lopencv_stitching249 -lopencv_ts249 -lopencv_video249 -lopencv_videostab249';  
if is_64bit  
    CPPFLAGS = [CPPFLAGS ' -largeArrayDims'];  
end  
%% add your files here!  
compile_files = {
    
    % the list of your code files which need to be compiled  
    'carplate.cpp'   
    
};  
   
%-------------------------------------------------------------------  
%% compiling...  
for k = 1 : length(compile_files)  
    str = compile_files{k};  
    fprintf('compilation of: %s\n', str);  
    str = [str ' -outdir ' out_dir CPPFLAGS LDFLAGS LIBS];  
    args = regexp(str, '\s+', 'split');  
    mex(args{:});  
end  
  
fprintf('Congratulations, compilation successful!!!\n');  