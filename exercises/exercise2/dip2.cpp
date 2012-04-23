//============================================================================
// Name        : dip2.cpp
// Author      : Till Rohrmann
// Version     : 0.1
// Copyright   : -
// Description : load image using opencv, apply noise reduction, save images
//============================================================================

#include <iostream>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// function headers of not yet implemented functions
void spatialConvolution(Mat&, Mat&, Mat&);
void averageFilter(Mat& src, Mat& dst, int kSize);
void medianFilter(Mat& src, Mat& dst, int kSize);
void adaptiveFilter(Mat& src, Mat& dst, int kSize, double threshold);
// function headers of given functions
void noiseReduction(Mat&, Mat&, const char*, int, int=0);
void generateNoisyImages(Mat&);

void flip(Mat& src, Mat &dst);

const int REF_KERNELSIZE=3;

// usage: argv[1] == "generate" to generate noisy images, path to original image in argv[2]
// 	  argv[1] == "restorate" to load and restorate noisy images
int main(int argc, char** argv) {

	// check whether parameter were specified
  	if (argc < 2){
	    cout << "Usage:\n\tdip2 generate path_to_original\n\tdip2 restorate"  << endl;
	    exit(-1);
	}
  
	// in a first step generate noisy images
	// path of original image is in argv[2]
	if (strcmp(argv[1], "generate") == 0){
	  
	  // check whether original image was specified
	  if (argc < 3){
	    cerr << "ERROR: original image not specified"  << endl;
	    exit(-1);
	  }
	  
	  // load image, path in argv[2], force gray-scale
	  cout << "load original image" << endl;
	  Mat img = imread(argv[2],0);
	  if (!img.data){
	    cerr << "ERROR: file " << argv[2] << " not found" << endl;
	    exit(-1);
	  }
	  // convert to floating point precision
	  img.convertTo(img,CV_32FC1);
	  cout << "done" << endl;
	  
	  // generate images with different types of noise
	  cout << "generate noisy images" << endl;
	  generateNoisyImages(img);
	  cout << "done" << endl;
	  cout << "Please run now: dip2 restorate" << endl;
	}

	// in a second step try to restorate noisy images
	if (strcmp(argv[1], "restorate") == 0){
	  
	  // some images
	  Mat img, orig, noise1, noise2, restorated1, restorated2;
  
	  // load images
	  cout << "load images" << endl;
	  orig = imread("original.jpg",0);
	  if (!orig.data){
	    cerr << "ERROR: original.jpg not found" << endl;
	    exit(-1);
	  }
	  orig.convertTo(orig, CV_32FC1);
	  noise1 = cvLoadImage("noiseType_1.jpg",0);
	  if (!noise1.data){
	    cerr << "noiseType_1.jpg not found" << endl;
	    exit(-1);
	  }
	  noise1.convertTo(noise1, CV_32FC1);
	  noise1.copyTo(restorated1);
	  noise2 = imread("noiseType_2.jpg",0);
	  if (!noise2.data){
	    cerr << "noiseType_2.jpg not found" << endl;
	    exit(-1);
	  }
	  noise2.convertTo(noise2, CV_32FC1);
	  noise2.copyTo(restorated2);
	  cout << "done" << endl;

	  //noise reduction
	  //Noise1 is a salt-and-pepper noise for which the median filter is best suited, because
	  //only few pixels are distorted. Applying an average filter would also distort the correct
	  //pixels, which is the reason for this filter choice.
	  noiseReduction(noise1, restorated1, "median", 7);
	  //Noise2 is a gaussian noise for which one should use the average filter or the adaptive filter.
	  //That's because the noise distribution cancels out if one calculates the average, since the mean is
	  //is 0. The best results could be achieved by using the adaptive filter.
	  noiseReduction(noise2, restorated2, "average", 5, 15);
	  
	  // save images
	  imwrite("restorated1.jpg", restorated1);
	  imwrite("restorated2.jpg", restorated2);

	}

	return 0;

}

// noise reduction
// src: input image
// dst: output image
// method: name of method to be performed
//	    "average" ==> moving average
//	    "median" ==> median filter
//	    "adaptive" ==> edge preserving average filter
// kSize: window size used by median operation
// thresh: threshold to be used only by adaptive average
void noiseReduction(Mat& src, Mat& dst, const char* method, int kSize, int thresh){

  // apply moving average filter
  if (strcmp(method, "average") == 0){
    averageFilter(src, dst, kSize);
  }
  // apply median filter
  if (strcmp(method, "median") == 0){
    medianFilter(src, dst, kSize);
  }
  // apply adaptive average filter
  if (strcmp(method, "adaptive") == 0){
    adaptiveFilter(src, dst, kSize, thresh);
  }

}

/**
 * This function performs a flip operation on a given matrix:
 * Input: src
 * Output: dst
 *
 * After the operation has finished the following holds:
 * 	(src)ij = (dst)(n-i)(m-j)
 * 	with n denoting the number of rows of src and m denoting the number of columns
 */
void flip(Mat& src,Mat &dst){
	assert(src.rows == dst.rows && src.cols == dst.cols);

	if(&src == &dst){
		for(int row=0; row < src.rows/2; row++){
			float* srcRow = src.ptr<float>(row);
			float* dstRow = dst.ptr<float>(src.rows-row -1);

			for(int col =0; col < src.cols; col++){
				float swap = dstRow[src.cols-col-1];
				dstRow[src.cols-col-1] = srcRow[col];
				srcRow[col] = swap;
			}
		}

		if(src.rows % 2 == 1){
			float* srcRow = src.ptr<float>(src.rows/2);
			for(int col = 0; col < src.cols/2;col++){
				float swap = srcRow[col];
				srcRow[col] = srcRow[src.cols-col-1];
				srcRow[src.cols-col-1] = swap;
			}
		}
	}else{
		for(int row =0; row < src.rows;row++){
			for(int col =0; col < src.cols; col++){
				dst.at<float>(src.rows-1-row,src.cols-1-col) = src.at<float>(row,col);
			}
		}
	}
}

// the average filter
// src: input image
// dst: output image
// kSize: window size used by local average
void averageFilter(Mat& src, Mat& dst, int kSize){
	assert((kSize % 2) ==1 );
	Mat kernel(kSize,kSize,CV_32FC1,1.0/(kSize*kSize));
	spatialConvolution(src,dst,kernel);
}

// the adaptive filter
// src: input image
// dst: output image
// kSize: window size used by local average
void adaptiveFilter(Mat& src, Mat& dst, int kSize, double threshold){
	// Intermediate matrix to hold 3x3 average filtering result
	Mat tmp(src.rows, src.cols, CV_32FC1);
	// Create filtered images
	averageFilter(src, tmp, REF_KERNELSIZE);
	averageFilter(src, dst, kSize);
	// Adaptively select larger filter if no edge
	for(int y = 0; y < src.rows; y++) {
		for(int x = 0; x < src.cols; x++) {
			if(abs(dst.at<float>(y,x) - tmp.at<float>(y,x)) > threshold)
				dst.at<float>(y,x) = src.at<float>(y,x);
		}
	}
}

// the median filter
// src: input image
// dst: output image
// kSize: window size used by median operation
void medianFilter(Mat& src, Mat& dst, int kSize){
	int cnt;

    for(int y = 0; y < src.rows; y++) {
		for(int x = 0; x < src.cols; x++) {
			cnt = 0;
			vector<float> tmp;
			for(int i = y-kSize/2; i <= y+kSize/2; i++)
				for(int j = x-kSize/2; j <= x+kSize/2; j++) {
					if(i > 0 && j > 0 && i < src.rows && j < src.cols) {
						tmp.push_back(src.at<float>(i,j));
						cnt++;
					}
				}
			sort(tmp.begin(), tmp.end());
			dst.at<float>(y,x) = tmp[cnt/2];
		}
	}

}

void spatialConvolution(Mat& in, Mat& out, Mat& kernel){
	Mat fkern;
	float sum;
	int ik, jk;

	kernel.copyTo(fkern);

    // First step: flip kernel about centre
    flip(kernel, fkern);


	// Now move across image and perform dot products
	for(int y = 0; y < in.rows; y++) {
		for(int x = 0;  x < in.cols; x++) {
			sum = 0.0;
			// Current pixel for filtering: (x,y)
			for(int i = y-fkern.rows/2, ik = 0; i <= y+fkern.rows/2; i++, ik++)
				for(int j = x-fkern.cols/2, jk = 0; j <= x+fkern.cols/2; j++, jk++){
					int row = i;
					int col = j;

					while(row <0){
						row += in.rows;
					}

					while(row >= in.rows){
						row -= in.rows;
					}

					while(col < 0){
						col += in.cols;
					}

					while(col >= in.cols){
						col -= in.cols;
					}

					sum += in.at<float>(row,col)*fkern.at<float>(ik,jk);
				}
			out.at<float>(y,x) = sum;
		}
	}
}

// generates and saves different noisy versions of input image
// orig: input image
void generateNoisyImages(Mat& orig){
 
    // save original
    imwrite("original.jpg", orig);
  
    // some temporary images
    Mat tmp1(orig.rows, orig.cols, CV_32FC1);
    Mat tmp2(orig.rows, orig.cols, CV_32FC1);
   
    // first noise operation
    float noiseLevel = 0.15;
    randu(tmp1, Scalar(0), Scalar(1));
    threshold(tmp1, tmp2, noiseLevel, 1, CV_THRESH_BINARY);
    tmp2 = tmp2.mul(orig);
    
    threshold(tmp1, tmp1, 1-noiseLevel, 1, CV_THRESH_BINARY);
    tmp1 *= 255;
    tmp1 = tmp2 + tmp1;
    threshold(tmp1, tmp1, 255, 255, CV_THRESH_TRUNC);
    // save image
    imwrite("noiseType_1.jpg", tmp1);
    
    // second noise operation
    noiseLevel = 10; //50;
    randn(tmp1, Scalar(0), Scalar(noiseLevel));
    tmp1 = orig + tmp1;
    threshold(tmp1,tmp1,255,255,CV_THRESH_TRUNC);
    threshold(tmp1,tmp1,0,0,CV_THRESH_TOZERO);
    // save image
    imwrite("noiseType_2.jpg", tmp1);

}
