#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <vector>
#include <iomanip>
#include "SVMCarPlate.h"

using namespace std;
using namespace cv;

SVMCarPlate::SVMCarPlate(){
	//set SVM parameter
	SVM_params.svm_type = CvSVM::C_SVC;
	SVM_params.kernel_type = CvSVM::LINEAR; //CvSVM::LINEAR;  
	SVM_params.degree = 0;
	SVM_params.gamma = 1;
	SVM_params.coef0 = 0;
	SVM_params.C = 1;
	SVM_params.nu = 0;
	SVM_params.p = 0;
	SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);
	
	//set SVM traindata
	string filename = "SVM.xml";
	FileStorage fs;
	fs.open(filename, FileStorage::READ);

	fs["TrainingData"] >> SVM_train_data;
	fs["labels"] >>  SVM_train_label;

	svmClassifier.train(SVM_train_data, SVM_train_label, Mat(), Mat(), SVM_params);
}

//predict car plate
int SVMCarPlate::predict(Mat input){
	Mat feature;
	//cvCvtColor(&input,&input, CV_BGR2GRAY);
	feature = ColorHistFeature(input);
	
	int response = svmClassifier.predict(feature);
	return response;
}

//color histogram
Mat SVMCarPlate::ColorHistFeature(Mat src){
	Mat colorhist;
	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split(src, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	colorhist.push_back(b_hist);
	colorhist.push_back(g_hist);
	colorhist.push_back(r_hist);
	colorhist = colorhist.reshape(1, 1);
	colorhist.convertTo(colorhist, CV_32FC1);

	return colorhist;
}