//---------------------------------------------------------------------------------------
// Exercise Digital Image Processing
// Homework 1
// Author: Till Rohrmann
// Matriculation number: 343756
// 
// Description: Program opens an image specified on the command line and performs an
// hough line detection. The result of the detection is saved as "result.jpg'
//---------------------------------------------------------------------------------------

#include <iostream>
#include <cstring>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// function that performs a hough line transformation to detect lines in an image
Mat doSomethingThatMyTutorIsGonnaLike(Mat& img){
	Mat gray,blurred,detectedEdges,result;
	int kernel_size = 3;
	int lowThreshold = 40;
	int highThreshold = 100;
	vector<Vec4i> lines;
		
	cvtColor(img,gray,CV_BGR2GRAY);
	
	blur(gray,blurred,Size(3,3));
	
	//detect edges in the image
	Canny(gray,detectedEdges,lowThreshold,highThreshold,kernel_size);
	
	//convert result of Canny into a color picture s.t. we can draw colored lines
	cvtColor(detectedEdges,result,CV_GRAY2BGR);
	
	//find lines in image
	HoughLinesP(detectedEdges,lines,1,CV_PI/180,150,0,0);
	
	for(int i =0; i< lines.size(); i++){
		//draw detected lines into the result image
		line(result,Point(lines[i][0],lines[i][1]),Point(lines[i][2],lines[i][3]),Scalar(0,0,255),2,CV_AA);
	}
	
	return result;
}

// usage: path to image in argv[1]
// main function, loads and saves image
int main(int argc, char** argv) {

	// check if image path was defined
	if (argc != 2){
	    cerr << "Usage: dip1 <path_to_image>" << endl;
	    return -1;
	}
	
	// window names
	string win1 = string ("Original image");
	string win2 = string ("Result");
  
	// some images
	Mat inputImage, outputImage;
  
	// load image as gray-scale, path in argv[1]
	cout << "load image" << endl;
	inputImage = imread( argv[1] );
	cout << "done" << endl;
	
	// check if image can be loaded
	if (!inputImage.data){
	    cerr << "ERROR: Cannot read file " << argv[1] << endl;
	    exit(-1);
	}

	// show input image
	namedWindow( win1.c_str() );
	imshow( win1.c_str(), inputImage );
	
	// do something (reasonable)
	outputImage = doSomethingThatMyTutorIsGonnaLike( inputImage );
	
	// show result
	namedWindow( win2.c_str() );
	imshow( win2.c_str(), outputImage );
	
	// save result
	imwrite("result.jpg", outputImage);
	
	// wait a bit
	waitKey(0);

	return 0;

}
