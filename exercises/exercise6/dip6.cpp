//============================================================================
// Name        : dip6.cpp
// Author      : 
// Version     : 0.1
// Copyright   : -
// Description : Loads image, calculates structure tensor, defines and plots interest points
//============================================================================

#include <iostream>
#include <list>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const string STR_GAUSSIAN = "gaussian";
const string STR_DEVXGAUSSIAN ="devxgaussian";
const string STR_DEVYGAUSSIAN = "devygaussian";

const double weightCoefficient = 1;
const double isotropyMin = 0.75;

void createKernel(Mat& kernel, double sigma, string name);
void nonMaxSuppression(Mat& img, Mat& out);
void showImage(Mat& img, const char* win, int wait, bool show=true, bool save=false);
void getInterestPoints(Mat& img, double sigma, vector<KeyPoint>& points);

// usage: path to image in argv[1], sigma of directional gradient in argv[2], default 0.5
int main(int argc, char** argv) {
  
    // load image as grayscale, path in argv[1]
    Mat img = imread(argv[1], 0);
    if (!img.data){
		cerr << "ERROR: Cannot load image from " << argv[1] << endl;
	}
    // convert U8 to 32F
    img.convertTo(img, CV_32FC1);
    //showImage(img, "original", 0);
    
    // this vector will contain interest points
    vector<KeyPoint> points;
    
    // define standard deviation of directional gradient
    double sigma;
    if (argc < 3)
      sigma = 0.5;
    else
      sigma = atof(argv[2]);
    // calculate interest points
    getInterestPoints(img, sigma, points);
    

    // plot result
    img = imread(argv[1]);
    drawKeypoints(img, points, img, Scalar(0,0,255), DrawMatchesFlags::DRAW_OVER_OUTIMG + DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    //plotInterestPoints(img, points, true, true);
    showImage(img, "keypoints", 0, true, true);
    return 0;

}

// creates a specific kernel
/*
kernel	the calculated kernel
kSize	size of the kernel
name	specifies which kernel shall be computed
*/
void createKernel(Mat& kernel, double sigma, string name){
	int kSize = 10*sigma;
	if(kSize % 2 ==0){
		kSize++;
	}
	kernel = Mat(kSize,kSize,CV_32FC1,(Scalar)0);

	if(name == STR_GAUSSIAN){
		for(int r=0, y=-kSize/2;r<kSize;r++,y++){
			for(int c = 0, x =-kSize/2; c <kSize; c++,x++){
				kernel.at<float>(r,c) = ((float)1)/2/M_PI/sigma/sigma*exp(-(x*x+y*y)/2/sigma/sigma);
			}
		}
	}
	else if(name == STR_DEVXGAUSSIAN){
		for(int r= 0, y=-kSize/2;r<kSize;r++,y++){
			for(int c = 0, x = -kSize/2; c < kSize; c++,x++){
				kernel.at<float>(r,c) = -((float)x)/2.0/M_PI/sigma/sigma/sigma/sigma*exp(-(x*x+y*y)/2/sigma/sigma);
			}
		}

	}else if(name ==STR_DEVYGAUSSIAN){
		for(int r= 0, y=-kSize/2;r<kSize;r++,y++){
			for(int c = 0, x = -kSize/2; c < kSize; c++,x++){
				kernel.at<float>(r,c) = -((float)y)/2.0/M_PI/sigma/sigma/sigma/sigma*exp(-(x*x+y*y)/2/sigma/sigma);
			}
		}
	}
}

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


// uses structure tensor to define interest points (foerstner)
void getInterestPoints(Mat& img, double sigma, vector<KeyPoint>& points){
	Mat gaussKernel;
	Mat devxGaussKernel;
	Mat devyGaussKernel;

	createKernel(gaussKernel,sigma,STR_GAUSSIAN);
	createKernel(devxGaussKernel,sigma,STR_DEVXGAUSSIAN);
	createKernel(devyGaussKernel,sigma,STR_DEVYGAUSSIAN);

	Mat gx;
	Mat gy;

	spatialConvolution(img,gx,devxGaussKernel);
	spatialConvolution(img,gy,devyGaussKernel);

//	showImage(gx,"dev x",1,true,false);
//	showImage(gy,"dev y",1,true,false);

	Mat gxgx = Mat(gx.rows,gx.cols,CV_32FC1);
	Mat gygy = Mat(gy.rows,gy.cols,CV_32FC1);
	Mat gxgy = Mat(gx.rows,gx.cols,CV_32FC1);

	for(int r= 0; r<gx.rows;r++){
		for(int c = 0; c <gx.cols;c++){
			gxgx.at<float>(r,c) = gx.at<float>(r,c)*gx.at<float>(r,c);
			gygy.at<float>(r,c) = gy.at<float>(r,c)*gy.at<float>(r,c);
			gxgy.at<float>(r,c) = gx.at<float>(r,c)*gy.at<float>(r,c);
		}
	}

//	showImage(gxgx,"gxgx",1,true,false);

	Mat avggxgx = Mat(gx.rows,gx.cols,CV_32FC1,(Scalar)0);
	Mat avggygy;
	Mat avggxgy;

	spatialConvolution(gxgx,avggxgx,gaussKernel);
	spatialConvolution(gygy,avggygy,gaussKernel);
	spatialConvolution(gxgy,avggxgy,gaussKernel);

//	showImage(avggxgx,"average gxgx",1,true,false);

	Mat trace = Mat(gx.rows,gx.cols,CV_32FC1);
	Mat det= Mat(gx.rows,gx.cols,CV_32FC1);

	for(int r =0; r< gx.rows;r++){
		for(int c=0; c<gx.cols;c++){
			trace.at<float>(r,c) = avggxgx.at<float>(r,c)+avggygy.at<float>(r,c);
			det.at<float>(r,c) = avggxgx.at<float>(r,c)*avggygy.at<float>(r,c) - avggxgy.at<float>(r,c)*avggxgy.at<float>(r,c);
		}
	}

//	showImage(trace,"trace",1,true,false);
//	showImage(det,"det",1,true,false);

	Mat weight = Mat(gx.rows,gx.cols,CV_32FC1,(Scalar)0);
	double averageWeight =0;

	for(int r = 0; r< gx.rows;r++){
		for(int c =0; c < gx.cols;c++){
			if(trace.at<float>(r,c) != 0){
				weight.at<float>(r,c) = det.at<float>(r,c)/trace.at<float>(r,c);
				averageWeight += weight.at<float>(r,c);
			}
		}
	}

	averageWeight /= gx.cols*gx.rows;

//	showImage(weight,"weight",1,true,false);

	nonMaxSuppression(weight,weight);

	Mat weightThr = Mat(gx.rows,gx.cols,CV_32FC1,(Scalar)0);

    cv::threshold(weight,weightThr,averageWeight*weightCoefficient,1,THRESH_BINARY);

//	showImage(weightThr,"weight threshold",1,true,false);

	Mat isotropy = Mat(gx.rows,gx.cols,CV_32FC1,(Scalar)0);

	for(int r= 0; r< gx.rows; r++){
		for(int c = 0; c < gx.cols; c++){
			if(trace.at<float>(r,c) != 0)
				isotropy.at<float>(r,c) = 4*det.at<float>(r,c)/(trace.at<float>(r,c)*trace.at<float>(r,c));
		}
	}

//	showImage(isotropy,"isotropy",1,true,false);
	nonMaxSuppression(isotropy,isotropy);

	Mat isotropyThr = Mat(gx.rows,gx.cols,CV_32FC1,(Scalar)0);

	cv::threshold(isotropy,isotropyThr,isotropyMin,1,THRESH_BINARY);

//	showImage(isotropyThr,"isotropy threshold",1,true,false);

	for(int r= 0; r< gx.rows; r++){
		for(int c = 0; c< gx.cols; c++){
			if(weightThr.at<float>(r,c) != 0 && isotropyThr.at<float>(r,c) != 0){
				KeyPoint point(c,r,1);
				points.push_back(point);
			}
		}
	}

}

// function to show and save one-channel images
void showImage(Mat& img, const char* win, int wait, bool show, bool save){
  
    Mat aux = img.clone();

    // scale and convert
    if (img.channels() == 1)
		normalize(aux, aux, 0, 255, CV_MINMAX);
    aux.convertTo(aux, CV_8UC1);
    // show
    if (show){
      imshow( win, aux);
      waitKey(wait);
    }
    // save
    if (save)
      imwrite( (string("img/") + string(win)+string(".png")).c_str(), aux);
}

// non-maxima suppression
// if any of the pixel at the 4-neighborhood is greater than current pixel, set it to zero
void nonMaxSuppression(Mat& img, Mat& out){

	Mat tmp = img.clone();
	
	for(int x=1; x<out.cols-1; x++){
		for(int y=1; y<out.rows-1; y++){
			if ( img.at<float>(y-1, x) >= img.at<float>(y, x) ){
				tmp.at<float>(y, x) = 0;
				continue;
			}
			if ( img.at<float>(y, x-1) >= img.at<float>(y, x) ){
				tmp.at<float>(y, x) = 0;
				continue;
			}
			if ( img.at<float>(y, x+1) >= img.at<float>(y, x) ){
				tmp.at<float>(y, x) = 0;
				continue;
			}
			if ( img.at<float>( y+1, x) >= img.at<float>(y, x) ){
				tmp.at<float>(y, x) = 0;
				continue;
			}
		}
	}
	tmp.copyTo(out);
}
