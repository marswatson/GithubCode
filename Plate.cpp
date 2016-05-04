#include "Plate.h"
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

using namespace std;
using namespace cv;

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
			if (seed.x <= 0)
				seed.x = 1;
			if (seed.x >= img_org.cols)
				seed.x = img_org.cols - 1;
			seed.y = rotated_rec[i].center.y + rand() % (int)minSize - (minSize / 2);
			if (seed.y <= 0)
				seed.y = 1;
			if (seed.y >= img_org.rows)
				seed.y = img_org.rows - 1;
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
	float ratio = (mr.size.height / mr.size.width) < 1 ? mr.size.width / mr.size.height: mr.size.height / mr.size.width;
	float area = float(mr.size.height * mr.size.width);

	//if the ratio of input rectangle doesn't satisfy the conditon return false
	if (ratio < standard_ratio * (1 - error) || ratio > standard_ratio * (1 + error)
		|| area < min_area || area > max_area)
		return false;
	else
		return true;
}
