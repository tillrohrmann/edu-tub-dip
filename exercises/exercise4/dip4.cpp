//============================================================================
// Name        : dip4.cpp
// Author      : 
// Version     : 0.1
// Copyright   : -
// Description : load image using opencv, apply unsharp masking
//============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

string STR_GAUSSIAN="gaussian";
string STR_AVERAGE="average";


// function headers of not yet implemented functions
void createKernel(Mat& kernel, int kSize, string name = "gaussian");
void circShift(Mat& in, Mat& out, int dx, int dy);
void frequencyConvolution(Mat& in, Mat& out, Mat& kernel);
void spatialConvolution(Mat& in, Mat& out, Mat& kernel);
void usm(Mat& in, Mat& out, int smoothType, int size, double thresh, double scale);
// function headers of given functions
void mySmooth(Mat& in, Mat& out, int size, string type, bool spatial);

// usage: path to image in argv[1]
int main(int argc, char** argv) {

	// save time measurements
	fstream fileSpatial;
	fstream fileFrequency;
	fileSpatial.open("convolutionSpatialDomain.txt", ios::out);
	fileFrequency.open("convolutionFrequencyDomain.txt", ios::out);
  
  	// some windows for displaying images
	const char* win_1 = "Original Image";
	const char* win_2 = "Enhanced Image";
	const char* win_3 = "Differences";
	namedWindow( win_1, CV_WINDOW_AUTOSIZE );
	namedWindow( win_2, CV_WINDOW_AUTOSIZE );
	namedWindow( win_3, CV_WINDOW_AUTOSIZE );
	
	// several images
	Mat imgIn, imgOut8U, imgOut32F, tmp;
  
	// for time measurements
	clock_t time;
	
	// parameter of USM
	int numberOfKernelSizes = 10;			// number of differently sized smoothing kernels
	// TO DO !!! try different values
	double thresh = 20;				// difference necessary to perform operation
	double scale = 2;				// scaling of edge enhancement

	// load image, path in argv[1]
	cout << "load image" << endl;
	imgIn = imread(argv[1]);
	// distort image
	int size = 3;
	GaussianBlur(imgIn, imgIn, Size(floor(size/2)*2+1,floor(size/2)*2+1), size/5., size/5.);
	cout << "done" << endl;

	// show original
	imshow( win_1, imgIn);

	// create output images
	imgOut8U  = Mat(imgIn.rows, imgIn.cols, CV_8UC3);
	imgOut32F = Mat(imgIn.rows, imgIn.cols, CV_32FC3);
	
	// convert and split input image
	// convert BGR to HSV
	cvtColor(imgIn, imgIn, CV_BGR2HSV);
	// convert U8 to 32F
	imgIn.convertTo(imgIn, CV_32FC3);
	// split into planes
	vector<Mat> planes;
	planes.push_back( Mat(imgIn.rows, imgIn.cols, CV_32FC1) );
	planes.push_back( Mat(imgIn.rows, imgIn.cols, CV_32FC1) );
	planes.push_back( Mat(imgIn.rows, imgIn.cols, CV_32FC1) );
	tmp = Mat(imgIn.rows, imgIn.cols, CV_32FC1);
	split(imgIn, planes);
	Mat value = planes.at(2).clone();

	// unsharp masking
	// try different kernel sizes
	for(int s=1; s<=numberOfKernelSizes; s++){
	  
	    // use this size for smoothing
	    int size = 4*s+1;

	    // either working in spatial or frequency domain
	    for(int type=0; type<2; type++){

		// speak to me
	      	switch(type){
		  case 0: cout << "> USM (" << size << "x" << size << ", using spatial domain):\t" << endl;break;
		  case 1: cout << "> USM (" << size << "x" << size << ", using frequency domain):\t" << endl;break;
		}
	      
		// measure starting time
		time = clock();
		// perform unsharp masking
		usm(value, tmp, type+1, size, thresh, scale);
		// measure stopping time
		time = (clock() - time);
		// print the ellapsed time
		switch(type){
		  case 0:{
		    cout << ((double)time)/CLOCKS_PER_SEC << "sec\n" << endl;
		    fileSpatial << ((double)time)/CLOCKS_PER_SEC << endl;
		  }break;
		  case 1:{
		    cout << ((double)time)/CLOCKS_PER_SEC << "sec\n" << endl;
		    fileFrequency << ((double)time)/CLOCKS_PER_SEC << endl;
		  }break;
		}
		
		// produce output image
		planes.at(2) = tmp;
		// merge planes to color image
		merge(planes, imgOut32F);
		// convert 32F to U8
		imgOut32F.convertTo(imgOut8U, CV_8UC3);
		// convert HSV to BGR
		cvtColor(imgOut8U, imgOut8U, CV_HSV2BGR);
		
		// show and save output images
		// create filename
		ostringstream fname;
		fname << string(argv[1]).substr(0,string(argv[1]).rfind(".")) << "_USM_" << size << "x" << size << "_";
		switch(type){
		  case 0: fname << "spatialDomain";break;
		  case 1: fname << "frequencyDomain";break;
		}
		imshow( win_2, imgOut8U);
// 		cvSaveImage((fname.str() + "_enhanced.jpg").c_str(), imgOut8U);

		// produce difference image
		planes.at(2) = tmp - value;
		normalize(planes.at(2), planes.at(2), 0, 255, CV_MINMAX);
		// merge planes to color image
		merge(planes, imgOut32F);
		// convert 32F to U8
		imgOut32F.convertTo(imgOut8U, CV_8UC3);
		// convert HSV to BGR
		cvtColor(imgOut8U, imgOut8U, CV_HSV2BGR);
		imshow( win_3, imgOut8U);
// 		cvSaveImage((fname.str() + "_diff2original.jpg").c_str(), imgOut8U);
		
		// images will be displayed for 3 seconds
		cvWaitKey(6000);
		// reset to original
		planes.at(2) = value;
	    }
	}

	// be tidy
	fileSpatial.close();
	fileFrequency.close();
	
	return 0;

}

/*
Generates a filter kernel of given size
kernel:		the generated filter kernel
kSize		the kernel size
type:		specifies type of filter, eg "gaussian", "average", ...
*/
void createKernel(Mat& kernel, int kSize, string name){
	assert(kSize%2==1);
  
	// some variables for gaussian filter kernel
	float sigma_x = kSize/5.0;	// standard deviation in x direction
	float sigma_y = kSize/5.0;	// standard deviation in y direction

	if(name==STR_GAUSSIAN){
		kernel = Mat::zeros(kSize, kSize, CV_32FC1);

		for(int r=0,vr=-kSize/2; r< kSize;r++,vr++){
			for(int c=0,vc=-kSize/2; c < kSize; c++,vc++){
				kernel.at<float>(r,c) = exp(-1.0/2*(vc*vc/(sigma_x*sigma_x)+vr*vr/(sigma_y*sigma_y)))/(2*M_PI*sigma_x*sigma_y);
			}
		}
	}else if(name ==STR_AVERAGE){
		kernel = Mat(kSize,kSize,CV_32FC1,(float)1.0/(kSize*kSize));
	}else{
		throw Exception();
	}
}

/*
Performes a circular shift in (dx,dy) direction
in	input matrix
out	circular shifted matrix
dx	shift in x-direction
dy	shift in y-direction
*/
void circShift(Mat& in, Mat& out, int dx, int dy){
	assert(&in != & out);
	out = in.clone();

	for(int r= 0; r < in.rows; r++){
		for(int c=0; c< in.cols; c++){
			out.at<float>(r,c) = in.at<float>((r+dy)%in.rows,(c+dx)%in.cols);
		}
	}
}

/*
Performes a convolution by multiplication in frequency domain
in		input image
out		output image
kernel		filter kernel
*/
void frequencyConvolution(Mat& in, Mat& out, Mat& kernel){
	Mat ft = Mat(in.rows,in.cols,CV_32FC1);
	Mat imgKernel = Mat(in.rows,in.cols,CV_32FC1);
	Mat ftKernel = Mat(in.rows,in.cols,CV_32FC1);
	Mat tmp = imgKernel(cv::Rect(0,0,kernel.rows,kernel.cols));
	kernel.copyTo(tmp);
	circShift(imgKernel,ftKernel,kernel.cols/2,kernel.rows/2);
	cv::dft(in,ft,0);
	cv::dft(ftKernel,ftKernel,0);
	cv::mulSpectrums(ft,ftKernel,ft,0);

	cv::dft(ft,out,DFT_INVERSE+DFT_SCALE);
}

/*
Performes a convolution by multiplication in spatial domain
in		input image
out		output image
kernel		filter kernel
*/
void spatialConvolution(Mat& in, Mat& out, Mat& kernel){
	out = in.clone();
	Mat fkernel = kernel.clone();
	cv::flip(kernel,fkernel,-1);
	for(int r =0; r < in.rows;r++){
		for(int c =0; c< in.cols;c++){
			double sum = 0;
			for(int kr=0,vr = r-fkernel.rows/2; kr < fkernel.rows;kr++,vr++){
				for(int kc = 0,vc = c - fkernel.cols/2; kc < fkernel.cols;kc++,vc++){
					while(vr < 0){
						vr += in.rows;
					}
					while(vr >= in.rows){
						vr -= in.rows;
					}
					while(vc < 0){
						vc += in.cols;
					}
					while(vc >=in.cols){
						vc -= in.cols;
					}
					sum += fkernel.at<float>(kr,kc)*in.at<float>(vr,vc);
				}
			}
			out.at<float>(r,c) = sum;
		}
	}
}



/*
Performs UnSharp Masking to enhance fine image structures
in		input image
out		enhanced image
smoothType	integer defining the type of smoothing
		ie. 0 <==> gaussian, spatial domain; 1 <==> gaussian, frequency domain
size		size of used smoothing kernel
thresh		minimal intensity difference to perform operation
scale		scaling of edge enhancement
*/
void usm(Mat& in, Mat& out, int smoothType, int size, double thresh, double scale){

	// some temporary image 
  	Mat tmp(in.rows, in.cols, CV_32FC1);
	
	// calculate edge enhancement
	// smooth original image
	switch(smoothType){
	  case 1:{
	    mySmooth(in, tmp, size, "gaussian", true);
	  }break;
	  case 2:{
	    mySmooth(in, tmp, size, "gaussian", false);
	  }break;
	  default:{
	    GaussianBlur(in, tmp, Size(floor(size/2)*2+1, floor(size/2)*2+1), size/5., size/5.);
	    break;
	  }
	}

	Mat diff(in.rows,in.cols,CV_32FC1);
	diff = in-tmp;
	cv::threshold(diff,diff,thresh,0,cv::THRESH_TOZERO);
	diff *= scale;
	out = in + diff;
}

/*
*************************
***   GIVEN FUNCTIONS ***
*************************
*/

/*
Performes smoothing operation by convolution
in		input image
out		smoothed image
size		size of filter kernel
type		type of filter kernel
spatial		true if convolution shall be performed in spatial domain, false otherwise
*/
void mySmooth(Mat& in, Mat& out, int size, string type, bool spatial){

    // create filter kernel
    Mat kernel;
    createKernel(kernel, size, type);
 
    // perform convoltion
    if (spatial)
	spatialConvolution(in, out, kernel);
    else
	frequencyConvolution(in, out, kernel);

}
