#include "Plate.h"
#include "OCR.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <iostream>  
#include <fstream> 
#include <string> 
#include <math.h>
#include <stdlib.h>
#include "SVMCarPlate.h"


using namespace std;
using namespace cv;


int main(){
	string filename;
	////////////////////////////////////One image operation///////////////////////////////////////////////
	//filename = "Cars2//Img5.jpg";
	filename = "Demo/Demo5.bmp";
	Plate car_plates;
	//input a image
	Mat input = imread(filename);
	//imshow("input image", input);
	//calculate the car plate
	car_plates.Set(input);
	//calculate the che characters
	vector<OCR> plate_character;
	system("del DemoResults\\*.bmp");
	//system("rmdir plate_results /s/q");
	for (int i = 0; i < car_plates.PlateSVM.size(); i++){
		OCR ocr_temp(car_plates.PlateSVM[i]);
		ocr_temp.Segment();
		plate_character.push_back(ocr_temp);
		for (int j = 0; j < ocr_temp.Chars.size(); j++){
			stringstream ss;
			ss << "DemoResults//plate" << i + 1 << "_" << j + 1 << ".bmp";
			//ss << "multi_plate_results//plate" << i + 1 << "_" << j + 1 << ".jpg";
			//imshow(ss.str(), ocr_temp.Chars[j]);
			imwrite(ss.str(), ocr_temp.Chars[j]);
		}
	}



	////////////////////////////////////extract training data//////////////////////////////////////////////
	//vector<Plate> plate;
	////input the Image
	//for (int i = 1; i <= 224; i++){
	//	cout << i << endl;
	//	char temp[20];
	//	sprintf(temp,"Cars2\\Img%d.jpg",i);
	//	filename = temp;
	//	Mat img = imread(filename);
	//	Plate p(img);
	//	plate.push_back(p);
	//}

	//store all the rectangle
	//vector<vector<Mat>> Traing_Data;
	//for (int i = 0; i < plate.size(); i++){
	//	vector<Mat> temp;
	//	for (int j = 0; j < plate[i].PlateResults.size(); j++)
	//		temp.push_back(plate[i].PlateResults[j]);
	//	Traing_Data.push_back(temp);
	//}
	//vector<Mat> Plate_Data;
	//for (int i = 0; i < Traing_Data.size(); i++){
	//	for (int j = 0; j < Traing_Data[i].size(); j++){
	//		cout << i << endl;
	//		stringstream ss(stringstream::in | stringstream::out);
	//		ss << "color_result2\\" << "result" << "_" << i << "_" << j << ".jpg";
	//		imwrite(ss.str(), Traing_Data[i][j]);
	//		Plate_Data.push_back(Traing_Data[i][j]);
	//	}
	//}

	////store SVM rectangle
	//for (int i = 0; i < plate.size(); i++)
	//	for (int j = 0; j < plate[i].PlateSVM.size(); j++){
	//	stringstream ss;
	//	ss << "color_result2_svm\\" << "plate_" << i << "_" << j << ".jpg";
	//	imwrite(ss.str(), plate[i].PlateSVM[j]);
	//	}

	////OCR
	//vector<OCR> ocr_plate;
	//for (int i = 0; i <= plate.size(); i++)
	//	for (int j = 0; j < plate[i].PlateSVM.size(); j++){
	//		Mat img;
	//		cvtColor(plate[i].PlateSVM[j], img, CV_BGR2GRAY);
	//		OCR ocr_temp(plate[i].PlateSVM[j]);
	//		ocr_temp.Segment();
	//		ocr_plate.push_back(ocr_temp);
	//}
	//
	////store the charaters
	//for (int i = 0; i < ocr_plate.size(); i++){
	//	char temp[30];
	//	for (int j = 0; j < ocr_plate[i].Chars.size(); j++){
	//		sprintf(temp, "report-plate//plate%d-%d.jpg", i,j);
	//		imwrite(temp, ocr_plate[i].Chars[j]);
	//	}
	//}

	/////////////////////////////////////end///////////////////////////////////////////

	///////////////////////////////////SVM/////////////////////////////////////////
	//vector<Mat> Plate_Data;
	//for (int i = 1; i <= 635; i++){
	//	cout << i << endl;
	//	char temp[20];
	//	sprintf(temp,"color_result2\\\\result%d.jpg",i);
	//	filename = temp;
	//	Mat img = imread(filename, CV_BGR2GRAY);
	//	Plate_Data.push_back(img);
	//}
	//vector<int> predict_result;
	//vector<Mat> car_plate;
	//SVMCarPlate svm_car_plate;
	//for (int i = 0; i < Plate_Data.size(); i++){
	//	if (svm_car_plate.predict(Plate_Data[i]) == 1)
	//		car_plate.push_back(Plate_Data[i]);
	//}
	//
	//for (int i = 0; i < car_plate.size(); i++){
	//	stringstream ss;
	//	ss << "report-plate//" << "SVM_" << i << ".jpg";
	//	imwrite(ss.str(), car_plate[i]);
	//}

	////////////////////////////////OCR Test/////////////////////////////////////////////
	//vector<OCR> ocr_plate;
	//for (int i = 0; i <= 0; i++){
	//	cout << i << endl;
	//	char temp[50];
	//	sprintf(temp, "report-plate\\SVM_%d.jpg", i);
	//	string filename = temp;
	//	Mat img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	//	OCR ocr_temp(img);
	//	ocr_temp.Segment();
	//	ocr_plate.push_back(ocr_temp);
	//}
	//
	////store the charaters
	//for (int i = 0; i < ocr_plate.size(); i++){
	//	char temp[30];
	//	for (int j = 0; j < ocr_plate[i].Chars.size(); j++){
	//		sprintf(temp, "report-plate//plate%d-%d.jpg", i,j);
	//		imwrite(temp, ocr_plate[i].Chars[j]);
	//	}
	//}

	///////////////////////////////////Test MultiCar Image///////////////////////////////////////////////
	//input the MultiCar Image
	//vector<Plate> plate;
	//for (int i = 7; i <= 13; i++){
	//	cout << i << endl;
	//	char temp[20];
	//	sprintf(temp,"MultiCar\\Img%d.jpg",i);
	//	filename = temp;
	//	Mat img = imread(filename);
	//	Plate p(img);
	//	plate.push_back(p);
	//}

	////store all the rectangle
	//vector<Mat> Plate_Data;
	//for (int i = 0; i < plate.size(); i++){
	//	for (int j = 0; j < plate[i].PlateResults.size(); j++)
	//		Plate_Data.push_back(plate[i].PlateResults[j]);
	//}

	//for (int i = 0; i < Plate_Data.size(); i++){
	//	//cout << i << endl;
	//	stringstream ss(stringstream::in | stringstream::out);
	//	ss << "MultiResult\\" <<"result" << "_" << i << ".jpg";
	//	imwrite(ss.str(), Plate_Data[i]);
	//}
	
	//vector<Mat> Plate_Data;
	//for (int i = 0; i <= 73; i++){
	//	cout << i << endl;
	//	char temp[20];
	//	sprintf(temp,"MultiResult\\result_%d.jpg",i);
	//	filename = temp;
	//	Mat img = imread(filename, CV_BGR2GRAY);
	//	Plate_Data.push_back(img);
	//}

	////svm predict
	////vector<int> predict_result;
	//vector<Mat> car_plate;
	//SVMCarPlate svm_car_plate;
	//for (int i = 0; i < Plate_Data.size(); i++){
	//	if (svm_car_plate.predict(Plate_Data[i]) == 1)
	//		car_plate.push_back(Plate_Data[i]);
	//}
	//
	//for (int i = 0; i < car_plate.size(); i++){
	//	stringstream ss(stringstream::in | stringstream::out);
	//	ss << "SVMMulticar\\SVMM_"<< i << ".jpg";
	//	imwrite(ss.str(), car_plate[i]);
	//}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	waitKey(0);
	return 0;
}

//