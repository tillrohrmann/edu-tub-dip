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

void quicksort(float* data, int left, int right);
void flip(Mat& src, Mat &dst);
void mergeMats(Mat& src1, Mat& src2, Mat& mask, Mat& dst);

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
	  //Noise1 is a salt-and-pepper noise for which the median filter is best suited.
	  noiseReduction(noise1, restorated1, "median", 7);
	  //Noise2 is a gaussian noise for which one should use the average filter or the adaptive filter. The best results
	  //could be achieved by using the adaptive filter.
	  noiseReduction(noise2, restorated2, "adaptive", 5, 15);
	  
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

void quicksort(float* data, int left, int right){
	if(right-left == 1){
		if(data[left] > data[right]){
			float swap = data[left];
			data[left] = data[right];
			data[right] = swap;
		}
	}
	else if(right-left > 1){
		int l = left+1;
		int r = right;
		float pivot = data[left];

		while(l< r){
			while(data[l] < pivot && l < right){
				l++;
			}
			while(data[r] >= pivot && r > left){
				r--;
			}

			if(r > l){
				float swap = data[r];
				data[r] = data[l];
				data[l] = swap;
			}
		}
		if(r != left){
			data[left] = data[r];
			data[r] = pivot;
		}

		quicksort(data,left,r-1);
		quicksort(data,r+1,right);
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

/**
 * This function merges to matrices according to a mask.
 *
 * Input: Mat src1, src2, mask
 * Output: Mat dst
 *
 * (dst)ij = (mask)ij == 1 ? (src1)ij : (src2)ij
 */
void mergeMats(Mat& src1, Mat& src2, Mat& mask, Mat&dst){
	assert(src1.rows == src2.rows && src1.cols==src2.cols);
	assert(src2.rows == mask.rows && src2.cols==mask.cols);
	assert(mask.rows == dst.rows && mask.cols == mask.cols);

	for(int row = 0; row < src1.rows; row++){
		float*dstData = dst.ptr<float>(row);
		float*src1Data = src1.ptr<float>(row);
		float*src2Data = src2.ptr<float>(row);
		float*maskData = mask.ptr<float>(row);
		for(int col =0; col < src1.cols; col++){
			if(maskData[col] == 1){
				dstData[col] = src1Data[col];
			}else{
				dstData[col] = src2Data[col];
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
void adaptiveFilter(Mat& src, Mat& dst, int kSize, double thresh){
	Mat avg3;
	Mat avgn;
	Mat mask(dst.rows,dst.cols,CV_32FC1);
	dst.copyTo(avg3);
	dst.copyTo(avgn);

	averageFilter(src,avg3,REF_KERNELSIZE);
	averageFilter(src,avgn,kSize);

	threshold(abs(avg3-avgn),mask,thresh,1,THRESH_BINARY);

	mergeMats(src,avgn,mask,dst);
}

// the median filter
// src: input image
// dst: output image
// kSize: window size used by median operation
void medianFilter(Mat& src, Mat& dst, int kSize){
	assert((kSize % 2) == 1);
	float* data = new float[kSize*kSize];
	int halfKS = kSize/2;
	for(int row = 0; row < src.rows; row++){
		float* rowData = dst.ptr<float>(row);
		for(int col = 0; col < src.cols; col++){
			int numData =0;

			int startRow = MAX(0,row-halfKS);
			int endRow = MIN(src.rows,row+halfKS+1);
			int startCol = MAX(0,col-halfKS);
			int endCol = MIN(src.cols,col+halfKS+1);

			for(int krow =startRow; krow< endRow;krow++){
				for(int kcol=startCol; kcol < endCol; kcol++){
					data[numData++] = src.at<float>(krow,kcol);
				}
			}

			quicksort(data,0,numData-1);

			rowData[col] = data[numData/2];
		}
	}
	delete [] data;
}

void spatialConvolution(Mat& in, Mat& out, Mat& kernel){
	assert((kernel.rows %2) == 1 && (kernel.cols%2) == 1);
	flip(kernel,kernel);
	int hks = kernel.rows/2;

	for(int row = 0; row < in.rows; row++){
		float* dataRow = out.ptr<float>(row);
		for(int col = 0; col < in.cols; col++){
			float sum=0;
			for(int krow = 0; krow < kernel.rows; krow++){
				for(int kcol = 0; kcol < kernel.cols;kcol++){
					int srow = row -hks + krow;
					int scol = col-hks+kcol;

					while(srow < 0){
						srow += in.rows;
					}
					while(srow >= in.rows){
						srow -= in.rows;
					}

					while(scol < 0){
						scol += in.cols;
					}

					while(scol >= in.cols){
						scol -= in.cols;
					}

					sum += in.at<float>(srow,scol)*kernel.at<float>(krow,kcol);
				}
			}
			dataRow[col] = sum;
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
