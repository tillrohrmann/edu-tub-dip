//============================================================================
// Name        : dip5.cpp
// Author      : 
// Version     : 0.1
// Copyright   : -
// Description : load image using opencv, degrade image, and apply inverse as well as wiener filter
//============================================================================

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

const double epsilon = 0.05;

Mat degradeImage(Mat& imgIn32F, Mat& degradedImg, double filterDev, double snr);
Mat inverseFilter(Mat& degraded, Mat& gaussKernel);
Mat wienerFilter(Mat& degraded, Mat& gaussKernel, double snr);
void circShift(Mat& in, Mat& out, int dx, int dy);

// usage: path to image in argv[1]
int main(int argc, char** argv) {
  
    // load image, path in argv[1]
    cout << "load image" << endl;
    Mat img = imread(argv[1], 0);
    // if not successfull, throw error and exit
    if (!img.data){
	cerr << "ERROR: file " << argv[1] << " not found" << endl;
	exit(-1);
    }
    // convert U8 to 32F
    img.convertTo(img, CV_32FC1);
    cout << " > done" << endl;

    // safe gray-scale version of original image
    imwrite( "original.png", img );
  
    // degrade image
    cout << "degrade image" << endl;
    // TO DO !!! --> try different values for smoothing and noise
    double filterDev = 3;
    double snr = 75;
    Mat degradedImg;
    Mat gaussKernel = degradeImage(img, degradedImg, filterDev, snr);
    cout << " > done" << endl;
    
    // safe degraded image
    imwrite( "degraded.png", degradedImg );
   
    // inverse filter
    cout << "inverse filter" << endl;
    Mat restoredImgInverseFilter = inverseFilter(degradedImg, gaussKernel);
    cout << " > done" << endl;
    
    // safe restored image
    imwrite( "restored_inverse.png", restoredImgInverseFilter );
    
    // wiener filter
    cout << "wiener filter" << endl;
    Mat restoredImgWienerFilter = wienerFilter(degradedImg, gaussKernel, snr);
    cout << " > done" << endl;
    
    // safe restored image
    imwrite( "restored_wiener.png", restoredImgWienerFilter );
    	
    return 0;

}

/*
Function applies inverse filter to restorate a degraded image
degraded	degraded input image
restored	restorated output image
filter		filter which caused degradation
*/
Mat inverseFilter(Mat& degraded, Mat& filter){
	assert(degraded.rows > filter.rows && degraded.cols > filter.cols);

	Mat iFilter = Mat(degraded.rows,degraded.cols,CV_32FC1,Scalar(0));
	Mat submatrix = iFilter(cv::Rect(0,0,filter.cols,filter.rows));
	Mat sFilter;
	Mat ft;
	Mat ftFilter;
	Mat original = Mat(degraded.rows,degraded.cols,CV_32FC2,Scalar(0,0));
	Mat result;

	filter.copyTo(submatrix);

	circShift(iFilter,sFilter,filter.cols/2,filter.rows/2);

	cv::dft(degraded,ft,DFT_COMPLEX_OUTPUT);
	cv::dft(iFilter,ftFilter,DFT_COMPLEX_OUTPUT);

	double max = 0;
	for(int row =0; row < ftFilter.rows;row++){
		for(int col = 0; col< ftFilter.cols;col++){
			double length = ftFilter.at<Vec2f>(row,col)[0]*ftFilter.at<Vec2f>(row,col)[0] + ftFilter.at<Vec2f>(row,col)[1]*ftFilter.at<Vec2f>(row,col)[1];
			if(length > max)
				max = length;
		}
	}

	max = sqrt(max);

	for(int row =0; row < ft.rows;row++){
		for(int col = 0; col < ft.cols; col++){
			double length = ftFilter.at<Vec2f>(row,col)[0]*ftFilter.at<Vec2f>(row,col)[0] + ftFilter.at<Vec2f>(row,col)[1]*ftFilter.at<Vec2f>(row,col)[1];

			if(sqrt(length) >= epsilon*max){
				original.at<Vec2f>(row,col)[0] = (ft.at<Vec2f>(row,col)[0]*ftFilter.at<Vec2f>(row,col)[0] + ft.at<Vec2f>(row,col)[1]*ftFilter.at<Vec2f>(row,col)[1])/length;
				original.at<Vec2f>(row,col)[1] = (ft.at<Vec2f>(row,col)[1]*ftFilter.at<Vec2f>(row,col)[0] - ft.at<Vec2f>(row,col)[0]*ftFilter.at<Vec2f>(row,col)[1])/length;
			}else{
				original.at<Vec2f>(row,col)[0] = ft.at<Vec2f>(row,col)[0]*epsilon*max;
				original.at<Vec2f>(row,col)[1] = ft.at<Vec2f>(row,col)[1]*epsilon*max;
			}
		}
	}

	cv::dft(original,result,DFT_INVERSE+DFT_SCALE+DFT_REAL_OUTPUT);

	threshold(result, result, 255, 255, CV_THRESH_TRUNC);
	threshold(result, result, 0, 0, CV_THRESH_TOZERO);

	return result;
}

/*
Function applies wiener filter to restorate a degraded image
degraded	degraded input image
restored	restorated output image
filter		filter which caused degradation
snr		signal to noise ratio of the input image
*/
Mat wienerFilter(Mat& degraded, Mat& filter, double snr){
	assert(degraded.rows > filter.rows && degraded.cols > filter.cols);

	Mat iFilter = Mat(degraded.rows,degraded.cols,CV_32FC1,Scalar(0));
	Mat submatrix = iFilter(cv::Rect(0,0,filter.cols,filter.rows));
	Mat sFilter;
	Mat ft;
	Mat ftFilter;
	Mat original = Mat(degraded.rows,degraded.cols,CV_32FC2,Scalar(0,0));
	Mat q = Mat(degraded.rows,degraded.cols,CV_32FC2,Scalar(0,0));
	Mat result;

	filter.copyTo(submatrix);

	circShift(iFilter,sFilter,filter.cols/2,filter.rows/2);

	cv::dft(degraded,ft,DFT_COMPLEX_OUTPUT);
	cv::dft(iFilter,ftFilter,DFT_COMPLEX_OUTPUT);

	double max = 0;
	for(int row =0; row < ftFilter.rows;row++){
		for(int col = 0; col< ftFilter.cols;col++){
			double length = ftFilter.at<Vec2f>(row,col)[0]*ftFilter.at<Vec2f>(row,col)[0] + ftFilter.at<Vec2f>(row,col)[1]*ftFilter.at<Vec2f>(row,col)[1];
			if(length > max)
				max = length;
		}
	}

	max = sqrt(max);

	for(int row =0; row < ft.rows;row++){
		for(int col = 0; col < ft.cols; col++){
			double length = ftFilter.at<Vec2f>(row,col)[0]*ftFilter.at<Vec2f>(row,col)[0] + ftFilter.at<Vec2f>(row,col)[1]*ftFilter.at<Vec2f>(row,col)[1];
			if(sqrt(length) >= epsilon*max){
				q.at<Vec2f>(row,col)[0] = ftFilter.at<Vec2f>(row,col)[0]/(length+1/snr/snr);
				q.at<Vec2f>(row,col)[1] = -ftFilter.at<Vec2f>(row,col)[1]/(length+1/snr/snr);
			}else{
				q.at<Vec2f>(row,col)[0] = ftFilter.at<Vec2f>(row,col)[0]/(length+1/snr/snr);
				q.at<Vec2f>(row,col)[1] = -ftFilter.at<Vec2f>(row,col)[1]/(length+1/snr/snr);
			}
		}
	}

	for(int row =0; row <ft.rows;row++){
		for(int col =0; col < ft.cols; col++){
			original.at<Vec2f>(row,col)[0] = q.at<Vec2f>(row,col)[0]*ft.at<Vec2f>(row,col)[0]-q.at<Vec2f>(row,col)[1]*ft.at<Vec2f>(row,col)[1];
			original.at<Vec2f>(row,col)[1] = q.at<Vec2f>(row,col)[0]*ft.at<Vec2f>(row,col)[1]+q.at<Vec2f>(row,col)[1]*ft.at<Vec2f>(row,col)[0];
		}
	}

	cv::dft(original,result,DFT_INVERSE+DFT_SCALE+DFT_REAL_OUTPUT);

	threshold(result, result, 255, 255, CV_THRESH_TRUNC);
	threshold(result, result, 0, 0, CV_THRESH_TOZERO);

	return result;
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
******************************
******  Given functions  *****
******************************
*/

/*
Function degrades a given image with gaussian blur and additive gaussian noise
img		input image
degradedImg	degraded output image
filterDev	standard deviation of kernel for gaussian blur
snr		signal to noise ratio for additive gaussian noise
return:		the used gaussian kernel
*/
Mat degradeImage(Mat& img, Mat& degradedImg, double filterDev, double snr){

    // calculate filter size
    int kSize = round(filterDev*3)*2 - 1;
	
    // smooth
    Mat gaussKernel = getGaussianKernel(kSize, filterDev, CV_32FC1);
    gaussKernel = gaussKernel * gaussKernel.t();
    filter2D(img, degradedImg, -1, gaussKernel);
    
    // add noise
    Scalar mean, stddev;
    meanStdDev(img, mean, stddev);
    Mat noise = Mat::zeros(img.rows, img.cols, CV_32FC1);
    randn(noise, 0, stddev.val[0]/snr);
    degradedImg = degradedImg + noise;
    threshold(degradedImg, degradedImg, 255, 255, CV_THRESH_TRUNC);
    threshold(degradedImg, degradedImg, 0, 0, CV_THRESH_TOZERO);

    return gaussKernel;
}
