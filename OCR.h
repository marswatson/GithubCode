#ifndef OCR_H
#define OCR_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace std;
using namespace cv;

class OCR{
public:
	OCR(Mat input);
	bool verify(Mat mr);
	void Segment();
	Mat ResizeChar(Mat input);
	Mat img_input;
	vector<Mat> Chars;
};











#endif
