#ifndef SVMCarPlate_H
#define SVMCarPlate_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>


using namespace std;
using namespace cv;

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

#endif;