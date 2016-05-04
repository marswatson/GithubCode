#include "OCR.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;

OCR::OCR(Mat input){
	img_input = input;
	cvtColor(input, img_input, CV_BGR2GRAY);
	//blur(img_gray,img_blur,Size(5,5));
}

void OCR::Segment(){
	Mat img_equalize;
	equalizeHist(img_input, img_equalize);
	//first we need get the binary picture
	Mat img_binary;
	threshold(img_equalize, img_binary, 75, 255, CV_THRESH_BINARY_INV);
	//imshow("binary img", img_binary);

	//Then find the countors in the image
	vector<vector <Point>> Contours;
	Mat img_black,img_findcontour;
	img_input.copyTo(img_black);
	img_binary.copyTo(img_findcontour);
	findContours(img_findcontour, Contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	Mat img_contour = img_input;
	cvtColor(img_contour, img_contour, CV_GRAY2RGB);
	drawContours(img_contour, Contours, -1, cv::Scalar(0,255,0), 1);
	//imshow("countours", img_contour);
	
	//extract the charater areas from image
	vector<vector <Point>>::iterator it_contours = Contours.begin();
	vector<int> Char_x,Char_y;
	while (it_contours != Contours.end()){
		//contours Circumscribed rectangle
		Rect rec = boundingRect(Mat(*it_contours));
		Mat ROI(img_binary, rec);
		if (verify(ROI)){
			//rectangle(img_input, rec, Scalar(0, 255, 0), 2);
			//imshow("a", img_input);
			ROI = ResizeChar(ROI);
			Chars.push_back(ROI);
			Char_x.push_back(rec.x);
			Char_y.push_back(rec.y);
		}
		it_contours++;
	}

	if (Chars.size() == 0)
		return;

	for (int i = 0; i < Chars.size() - 1; i++)
		for (int j = i + 1; j < Chars.size();){
		float distance = sqrt(pow(Char_x[i] - Char_x[j], 2) + pow(Char_y[i] - Char_y[i], 2));
		if (distance < 2){
			Chars.erase(Chars.begin() + j);
			Char_x.erase(Char_x.begin() + j);
			Char_y.erase(Char_y.begin() + j);
		}
		else
			j++;
		}

	//sort by x coordinates
	for (int i = 0; i < Chars.size() - 1; i++){
		int max = Char_x[i];
		int maxInd = i;
		int j;
		for (j = i + 1; j < Chars.size(); j++){
			if (max > Char_x[j]){
				max = Char_x[j];
				maxInd = j;
			}
		}
		Mat temp = Chars[maxInd];
		Char_x[maxInd] = Char_x[i];
		Chars[maxInd] = Chars[i];
		Char_x[i] = max;
		Chars[i] = temp;
	}
	//waitKey(0);

}

bool OCR::verify(Mat r){
	//Char sizes 5x12  
	float aspect = 5.0f / 12.0f;
	float charAspect = (float)r.cols / (float)r.rows;
	float error = 0.45;
	float minHeight = 27;
	float maxHeight = 65;
	//We have a different aspect ratio for number 1, and it can be ~0.2  
	float minAspect = aspect - aspect*error;
	float maxAspect = aspect + aspect*error;
	//area of pixels  
	float area = countNonZero(r);
	//bb area  
	float bbArea = r.cols*r.rows;
	//% of pixel in area  
	float percPixels = area / bbArea;

	if (charAspect > minAspect && charAspect < maxAspect &&  r.rows >= minHeight && r.rows < maxHeight && percPixels < 0.8 && percPixels > 0.15)
		return true;
	else
		return false;
}

//use warp affine to resize the picture
Mat OCR::ResizeChar(Mat input){
	int h = input.rows;
	int w = input.cols;
	int charsize = 32;
	Mat transformMat = Mat::eye(2, 3, CV_32F);
	int m = max(w, h);
	transformMat.at<float>(0, 2) = m / 2 - w / 2;
	transformMat.at<float>(1, 2) = m / 2 - h / 2;
	Mat warpImage(m, m, input.type());
	warpAffine(input, warpImage, transformMat, warpImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

	Mat output;
	resize(warpImage, output, Size(charsize, charsize));
	return output;
}