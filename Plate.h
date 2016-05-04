#ifndef PLATE_H
#define PLATE_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SVMCarPlate.h"

using namespace std;
using namespace cv;


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
	Mat img_gray,img_org;
};

#endif