#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <stdlib.h>
#include <vector>
#include <iostream>  
#include <fstream> 
#include <string> 
#include <math.h>
#include <time.h>
#include <math.h>
#include "mex.h"


using namespace std;
using namespace cv;


///////////////////////////////////////////SVM car plate begin///////////////////////////////////////////////////////////
class SVMCarPlate {
public:
	SVMCarPlate();
	int predict(Mat input);
private:
	Mat SVM_train_data;
	Mat SVM_train_label;
	CvSVMParams SVM_params;
	////Train SVM  
	CvSVM svmClassifier;
	Mat ColorHistFeature(Mat);
};

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
	fs["labels"] >> SVM_train_label;

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
///////////////////////////////////////////SVM car plate end//////////////////////////////////////////////////////////

/////////////////////////////////////////PLate begin/////////////////////////////////////////////////////////////////
class Plate{
public:
	Plate();
	Plate(Mat img);
	void Set(Mat img);
	void PlateDetection();
	bool verifySizes(RotatedRect mr);
	vector<Mat > PlateResults;
	vector<Point2f> PlateCenters;
	vector<Mat> PlateSVM;
private:
	SVMCarPlate svm_car_plate;
	vector<Rect> position;
	Mat img_gray, img_org;
};

//create empty class
Plate::Plate(){

}
//create class with input image
Plate::Plate(Mat img){
	img_org = img;
	cvtColor(img, img_gray, CV_BGR2GRAY);
	PlateDetection();
}
void Plate::Set(Mat img){
	img_org = img;
	cvtColor(img, img_gray, CV_BGR2GRAY);
	PlateDetection();
}
//detect all the possible 
void Plate::PlateDetection(){

	//blur the image to avoid the noise effect
	Mat img_blur;
	blur(img_gray, img_blur, Size(5, 5));
	//imshow("blur image",img_blur);

	//use sobel to detect the vertical edge
	Mat img_sobel;
	Sobel(img_blur, img_sobel, CV_8U, 1, 0);
	//imshow("sobel image", img_sobel);

	//convert sobel image to a binary image with only 0 and 255
	Mat img_threshold;
	threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	//imshow("threshold image", img_threshold);

	//morphologic close operation to extract rectangle
	Mat img_morphologic;
	Mat element = getStructuringElement(MORPH_RECT, Size(14, 7));
	morphologyEx(img_threshold, img_morphologic, CV_MOP_CLOSE, element);
	//imshow("morphorlogic image",img_morphologic);

	//find the countour from the image after morphologic operation
	vector < vector<Point> > contours;
	findContours(img_morphologic, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	//remove the rectangle not satified with ratio and area
	vector< RotatedRect > rotated_rec;
	vector< vector<Point> >::iterator it = contours.begin();
	while (it != contours.end()){
		RotatedRect mr = minAreaRect(Mat(*it));
		if (!verifySizes(mr)){
			it = contours.erase(it);
		}
		else{
			it++;
			rotated_rec.push_back(mr);
		}
	}

	//draw contours
	Mat img_contours;
	img_org.copyTo(img_contours);
	drawContours(img_contours, contours, -1, Scalar(0, 255, 0), 3);
	//imshow("image with contours",img_contours);

	//extract rectangle according to the contours
	for (int i = 0; i < rotated_rec.size(); i++){
		//get the min size between width and height  
		float minSize = (rotated_rec[i].size.width < rotated_rec[i].size.height)
			? rotated_rec[i].size.width : rotated_rec[i].size.height;
		minSize = minSize - minSize*0.5;
		//initialize rand and get 5 points around center for floodfill algorithm  
		srand(time(NULL));
		//Initialize floodfill parameters and variables  
		Mat mask;
		mask.create(img_gray.rows + 2, img_gray.cols + 2, CV_8UC1);
		mask = Scalar::all(0);
		int loDiff = 30;
		int upDiff = 30;
		int connectivity = 8;
		int newMaskVal = 255;
		int NumSeeds = 10;
		Rect ccomp;
		int flags = connectivity + (newMaskVal << 8) + CV_FLOODFILL_FIXED_RANGE + CV_FLOODFILL_MASK_ONLY;
		for (int j = 0; j < NumSeeds; j++){
			Point2f seed;
			seed.x = rotated_rec[i].center.x + rand() % (int)minSize - (minSize / 2);
			if (seed.x <= 1)
				seed.x = 2;
			if (seed.x >= (img_org.cols-1))
				seed.x = img_org.cols - 2;
			seed.y = rotated_rec[i].center.y + rand() % (int)minSize - (minSize / 2);
			if (seed.y <= 1)
				seed.y = 2;
			if (seed.y >= (img_org.rows-1))
				seed.y = img_org.rows - 2;
			floodFill(img_gray, mask, seed, Scalar(255), &ccomp, Scalar(loDiff), Scalar(upDiff), flags);
		}
		//imshow("mask", mask);

		//extract patches points from the mask
		vector< Point> interest_points;
		Point temp_point;
		Mat_<uchar>::iterator it_mask = mask.begin<uchar>();
		Mat_<uchar>::iterator it_end = mask.end<uchar>();
		for (; it_mask != it_end; ++it_mask)
			if (*it_mask == 255)
				interest_points.push_back(it_mask.pos());

		//get the rotated rectangle from the patches points
		RotatedRect minRect = minAreaRect(interest_points);

		if (verifySizes(minRect)){
			Point2f rect_vertices[4];
			minRect.points(rect_vertices);

			//get the rectangle from the rotated rectangle
			//first we need to find rotation matrix
			//Get rotation matrix  
			float r = (float)minRect.size.width / (float)minRect.size.height;
			float angle = minRect.angle;
			if (r < 1)
				angle = 90 + angle;
			Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);
			//store the plate centers
			PlateCenters.push_back(minRect.center);

			//Second, create and rotate image  
			Mat img_rotated;
			//warpAffine(img_gray, img_rotated, rotmat, img_gray.size(), CV_INTER_CUBIC);
			warpAffine(img_org, img_rotated, rotmat, img_gray.size(), CV_INTER_CUBIC);

			//extract rectangle 
			Size rect_size = minRect.size;
			if (r < 1)
				swap(rect_size.width, rect_size.height);
			Mat img_crop;
			getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);

			Mat resultResized;
			resultResized.create(75, 150, CV_8UC3);
			resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);
			//equalizeHist(resultResized,resultResized);
			PlateResults.push_back(resultResized);
		}
	}
	//erase the repeat car plate
	if (PlateResults.size() == 0){
		Mat rec(75, 150, CV_8UC3, 1);
		PlateResults.push_back(rec);
		return;
	}
	vector<Point2f>::iterator center_iter = PlateCenters.begin();
	vector<Mat>::iterator plate_iter = PlateResults.begin();
	for (int i = 0; i < PlateCenters.size() - 1; i++)
		for (int j = i + 1; j < PlateCenters.size();){
		float distance = sqrt(pow(PlateCenters[i].x - PlateCenters[j].x, 2) + pow(PlateCenters[i].y - PlateCenters[j].y, 2));
		if (distance < 20){
			PlateCenters.erase(PlateCenters.begin() + j);
			PlateResults.erase(PlateResults.begin() + j);
		}
		else
			j++;
		}


	//Mat aa = PlateResults[0];
	//Mat bb = PlateResults[1];
	//Mat cc = PlateResults[2];
	////Mat dd = PlateResults[3];
	//svm predict
	for (int i = 0; i < PlateResults.size(); i++){
		if (svm_car_plate.predict(PlateResults[i]) == 1)
			PlateSVM.push_back(PlateResults[i]);
	}
	//Mat dd = PlateSVM[0];
	//int a = 1;
}



//verify the rectangle
bool Plate::verifySizes(RotatedRect mr){
	//set error rate
	float error = 0.5;

	//Ratio of width to height
	float width = 12;
	float height = 6;
	float standard_ratio = width / height;

	//area that satified the car plate
	float min_area = 30 * 30 * standard_ratio;
	float max_area = 180 * 180 * standard_ratio;

	//input rectangle ratio of width to height
	float ratio = (mr.size.height / mr.size.width) < 1 ? mr.size.width / mr.size.height : mr.size.height / mr.size.width;
	float area = float(mr.size.height * mr.size.width);

	//if the ratio of input rectangle doesn't satisfy the conditon return false
	if (ratio < standard_ratio * (1 - error) || ratio > standard_ratio * (1 + error)
		|| area < min_area || area > max_area)
		return false;
	else
		return true;
}
/////////////////////////////////////////////Plate end////////////////////////////////////////////////////////////

////////////////////////////////////////////OCR Begin///////////////////////////////////////////////////////////////
class OCR{
public:
	OCR();
	OCR(Mat input);
	bool verify(Mat mr);
	void Segment();
	Mat ResizeChar(Mat input);
	Mat img_input;
	vector<Mat> Chars;
};

OCR::OCR(){

}
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
	Mat img_black, img_findcontour;
	img_input.copyTo(img_black);
	img_binary.copyTo(img_findcontour);
	findContours(img_findcontour, Contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	Mat img_contour = img_input;
	cvtColor(img_contour, img_contour, CV_GRAY2RGB);
	drawContours(img_contour, Contours, -1, cv::Scalar(0, 255, 0), 1);
	//imshow("countours", img_contour);

	//extract the charater areas from image
	vector<vector <Point>>::iterator it_contours = Contours.begin();
	vector<int> Char_x, Char_y;
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
	float minAspect = 0.2;
	float maxAspect = aspect + aspect*error;
	//area of pixels  
	float area = countNonZero(r);
	//bb area  
	float bbArea = r.cols*r.rows;
	//% of pixel in area  
	float percPixels = area / bbArea;

	//We have a different aspect ratio for number 1, and it can be ~0.2  
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
/////////////////////////////////////////////////////OCR end////////////////////////////////////////////////////////

///////////////////////////////////////////////main function////////////////////////////////////////////////////////
void exit_with_help()
{
	mexPrintf(
		"Usage: [imageMatrix] = DenseTrack('imageFile.jpg');\n"
		);
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

void capstone(char *filename, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	// Get the name of the image  
	Plate car_plates;
	Mat image = imread(filename);
	if (image.empty()) {
		mexPrintf("can't open input file %s\n", filename);
		fake_answer(plhs);
		return;
	}

	//input a image
	Mat input = imread(filename);
	//calculate the car plate
	car_plates.Set(input);
	//calculate the che characters
	vector<OCR> plate_character;
	system("del matlabinput\\*.bmp");
	for (int i = 0; i < car_plates.PlateSVM.size(); i++){
		OCR ocr_temp(car_plates.PlateSVM[i]);
		ocr_temp.Segment();
		plate_character.push_back(ocr_temp);
		for (int j = 0; j < ocr_temp.Chars.size(); j++){
			stringstream ss;
			ss << "matlabinput//plate" << i + 1 << "_" << j + 1 << ".bmp";
			//ss << "multi_plate_results//plate" << i + 1 << "_" << j + 1 << ".jpg";
			//imshow(ss.str(), ocr_temp.Chars[j]);
			imwrite(ss.str(), ocr_temp.Chars[j]);
		}
	}
	return;
}




void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs == 1)
	{
		char filename[256];
		mxGetString(prhs[0], filename, mxGetN(prhs[0]) + 1);
		if (filename == NULL)
		{
			mexPrintf("Error: filename is NULL\n");
			exit_with_help();
			return;
		}

		capstone(filename, plhs, nrhs, prhs);
	}
	else
	{
		exit_with_help();
		fake_answer(plhs);
		return;
	}
}
//int main(){
//	string filename;
//	////////////////////////////////////One image operation///////////////////////////////////////////////
//	filename = "Cars2//Img5.jpg";
//	//filename = "MultiCar//Img1.jpg";
//	Plate car_plates;
//	//input a image
//	Mat input = imread(filename);
//	//imshow("input image", input);
//	//calculate the car plate
//	car_plates.Set(input);
//	//calculate the che characters
//	vector<OCR> plate_character;
//	for (int i = 0; i < car_plates.PlateSVM.size(); i++){
//		OCR ocr_temp(car_plates.PlateSVM[i]);
//		ocr_temp.Segment();
//		plate_character.push_back(ocr_temp);
//		for (int j = 0; j < ocr_temp.Chars.size(); j++){
//			stringstream ss;
//			ss << "plate_results//plate" << i + 1 << "_" << j + 1 << ".jpg";
//			//ss << "multi_plate_results//plate" << i + 1 << "_" << j + 1 << ".jpg";
//			//imshow(ss.str(), ocr_temp.Chars[j]);
//			imwrite(ss.str(), ocr_temp.Chars[j]);
//		}
//	}
//}