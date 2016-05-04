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

using namespace std;
using namespace cv;

Mat GreyHistFeature(Mat input);
Mat ColorHistFeature(Mat src);

int main(){

	////////////////////////////calculate the input image grey histogram/////////////////////////////////
	char temp[100];
	string filename;

	////SVM training data and training label
	Mat SVM_train_data;
	Mat SVM_train_label;
	Mat features;
	//read label 1 images and calculate the image histogram
	for (int i = 1; i <= 88; i++){
		sprintf(temp, "Testing\\Test3\\label1\\Img%d.jpg", i);
		filename = temp;
		Mat src = imread(filename);
		/// Compute the features:
		features = GreyHistFeature(src);
		SVM_train_data.push_back(features);
		SVM_train_label.push_back(1);
	}
	//read lebel 0 images and calculaate the image histogram
	for (int i = 1; i <= 88; i++){
		sprintf(temp, "Testing\\Test3\\label0\\Img%d.jpg", i); //\Training\Train1
		filename = temp;
		Mat src = imread(filename);

		/// Compute the histograms:
		features = GreyHistFeature(src);
		SVM_train_data.push_back(features);
		SVM_train_label.push_back(0);
	}

	SVM_train_data.convertTo(SVM_train_data, CV_32FC1);
	FileStorage fs("SVM.xml", FileStorage::WRITE);
	fs << "TrainingData" << SVM_train_data;
	fs << "labels" << SVM_train_label;
	fs.release();


	//FileStorage fs("SVM.xml", FileStorage::READ);
	//fs["TrainingData"] >> SVM_train_data;
	//fs["labels"] >> SVM_train_label;
	//fs.release();

	///////////////////////////draw histogram////////////////////////////////////////////////
	//// Draw the histograms for B, G and R
	//int hist_w = 512; int hist_h = 400;
	//int bin_w = cvRound((double)hist_w / histSize);
	//Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0));
	///// Normalize the result to [ 0, histImage.rows ]
	//normalize(grey_hist, grey_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	///// Draw for each channel
	//for (int i = 1; i < histSize; i++)
	//{
	//	line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(grey_hist.at<float>(i - 1))),
	//		Point(bin_w*(i), hist_h - cvRound(grey_hist.at<float>(i))),
	//		Scalar(255), 2, 8, 0);
	//}
	///// Display
	//namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	//imshow("calcHist Demo", histImage);
	////////////////////////////////////////////end///////////////////////////////////////

	//Setting up SVM parameters
	CvSVMParams SVM_params;
	SVM_params.svm_type = CvSVM::C_SVC;
	SVM_params.kernel_type = CvSVM::LINEAR; //CvSVM::LINEAR;  
	SVM_params.degree = 0;
	SVM_params.gamma = 1;
	SVM_params.coef0 = 0;
	SVM_params.C = 1;
	SVM_params.nu = 0;
	SVM_params.p = 0;
	SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);
	//Train SVM  
	CvSVM svmClassifier(SVM_train_data, SVM_train_label, Mat(), Mat(), SVM_params);

	//test the error rate for plate detectcion
	Mat SVM_test_label;
	Mat SVM_test_data;
	Mat groundTruth(0, 1, CV_32FC1);
	int count1 = 0, count2 = 0;
	//int a = 0, b = 0;
	for (int i = 1; i <= 96; i++){
		sprintf(temp, "Training\\Train3\\label1\\Img%d.jpg", i);
		filename = temp;
		Mat test = imread(filename);

		//equalizeHist(test, test);
		features = GreyHistFeature(test);
		SVM_test_data.push_back(features);
		int response = svmClassifier.predict(features);
		SVM_test_label.push_back(response);
		groundTruth.push_back(1);
		if (response != 1)
			count1++;
		cout << "Img " << i << " correct label is 1, svm predict label is: " << response << endl;
	}

	for (int i = 1; i <= 96; i++){
		sprintf(temp, "Training\\Train3\\label0\\Img%d.jpg", i);
		filename = temp;
		Mat test = imread(filename);

		features = GreyHistFeature(test);
		int response = svmClassifier.predict(features);
		SVM_test_data.push_back(features);
		SVM_test_label.push_back(response);
		groundTruth.push_back(0);
		if (response != 0)
			count2++;
		cout << "Img " << i << " correct label is 0, svm predict label is: " << response << endl;
	}


	double errorRate;
	//calculate the number of unmatched classes  
	errorRate = (double)countNonZero(groundTruth - SVM_test_label) / SVM_test_data.rows;
	cout << "Total error rate is: " << errorRate << endl;
	cout << "label 1 -> 0 count is " << count1 << "/96 error rate is " << double(count1) / 96 << endl;
	cout << "label 0 -> 1 count is " << count2 << "/96 error rate is " << double(count2) / 96 << endl;
	waitKey(0);
}

//grey histogram
Mat GreyHistFeature(Mat input){
	Mat temp;
	cvtColor(input,temp,CV_BGR2GRAY);
	/// Establish the number of bins
	int histSize = 64;
	/// Set the ranges
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	Mat grey_hist;

	calcHist(&temp, 1, 0, Mat(), grey_hist, 1, &histSize, &histRange, uniform, accumulate);
	grey_hist = grey_hist.reshape(1, 1);
	grey_hist.convertTo(grey_hist, CV_32FC1);
	return grey_hist;
}

//color histogram
Mat ColorHistFeature(Mat src){
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
