//============================================================================
// Name        : dip3.cpp
// Author      : Ronny Haensch
// Version     : 0.1
// Copyright   : -
// Description : textons for image retrieval
//============================================================================

#include <iostream>
#include <fstream>

#include <algorithm>
#include <vector>

#include <cassert>
#include <cmath>
#include <limits>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "IllegalArgumentException.h"

using namespace std;
using namespace cv;

const string STR_GAUSSIAN="gaussian";
const string STR_GAUSSIANDEVX="gaussianDevX";
const string STR_GAUSSIANDEVXX="gaussianDevXX";
const string STR_MEXHAT="mexHat";

// function headers of not yet implemented functions
void createKernel1D(Mat& kernel, int kSize, string name);
void spatialConvolution(Mat&, Mat&, Mat&, double);
void applyFilterbank(vector<Mat>& db, vector<Mat>& filterResp);
void getTextonImages(vector<Mat>& filterResp, Mat& textons, vector<Mat>& textonImages);
void calcTextonHistograms(vector<Mat>& textonImages, Mat& textonHistogram);
void findQuery(Mat& textonHistogram, Mat&db);
// function headers of given functions
int loadDB(vector<Mat>& db, string fname, int numberOfImages);
void distortQuery(vector<Mat>& queries);
void clustering(vector<Mat>& filterResp, Mat& textons, int numberOfDataPoints = (int)pow(10.0,3));
Mat loadMat(string fname);
void saveMat(Mat matrix, string fname);
bool isCommentline(string* buffer);
void readInitFile(string fname, int& numberOfTextons, string& imgDBFile, string& queryFile, string& textonsFile, string& databaseDescrFile, double& numberOfDataPoints, int& numberOfDBImages, int& numberOfQueryImages);

// usage: argv[1] == "generate" to calculate textons and texon-based image descriptors
// 	  argv[1] == "find" to load image database descriptors, calculate query image descriptors, and match them
//	  argv[2] ==  path to init file
int main(int argc, char** argv) {

	// check whether parameter were specified
  	if (argc < 3){
	    cout << "Usage:\n\tdip3 generate <init-file>\n\tdip3 find <init-file>"  << endl;
	    exit(-1);
	}
	// check proper usage
	if ((strcmp(argv[1], "generate") != 0) and (strcmp(argv[1], "find") != 0)){
	    cout << "Usage:\n\tdip3 generate <init-file>\n\tdip3 find <init-file>"  << endl;
	    exit(-1);
	}

	// system parameters, specified in init-file
	int numberOfTextons;					// number of textons to be used
	int numberOfDBImages, numberOfQueryImages;		// for debugging purposes one might want to constrain the number of images
	string imgDBFile, queryFile;				// paths to load database and query images
	string textonsFile, databaseDescrFile;			// paths to save/load calculated textons and image descriptors
	double numberOfDataPoints;				// number of data points used during clustering

	// read init file
	// TO DO !!! --> check and edit init file (if necessary...)
	readInitFile(argv[2], numberOfTextons, imgDBFile, queryFile, textonsFile, databaseDescrFile, numberOfDataPoints, numberOfDBImages, numberOfQueryImages);
	// check for correct usage of init file
	if (numberOfTextons*numberOfDBImages*numberOfQueryImages*numberOfDataPoints*textonsFile.length()*databaseDescrFile.length() == 0){
	    cout << "ERROR: wrong parameter settings in init file"  << endl;
	    exit(-1);
	}
    
	// to calulate textons and generate image descriptors for the database
	if (strcmp(argv[1], "generate") == 0){
  
	    // load images as gray-scale
	    cout << "load image database" << endl;
	    vector<Mat> db;
	    numberOfDBImages = loadDB(db, imgDBFile, numberOfDBImages);
	    cout << "done: There are " << db.size() << " images" << endl;
	  
	    // apply filterbank to all images to get filter responses
	    cout << "apply filterbank" << endl;
	    vector<Mat> filterResp;
	    applyFilterbank(db, filterResp);
	    cout << "done" << endl;
	    // original images are not needed anymore
	    db.clear();
	  
	    // cluster the filter responses, cluster centers correspond to textons
	    cout << "cluster filter responses" << endl;
	    Mat textons = Mat::zeros(numberOfTextons, 8, CV_32FC1);
	    clustering(filterResp, textons, numberOfDataPoints);
	    cout << "done" << endl;
	    // save textons
	    saveMat(textons, textonsFile);
	    
	    // calculate the distance of each pixel (ie. filter response) to each texton
	    cout << "get texton images" << endl;
	    vector<Mat> textonImages;
	    getTextonImages(filterResp, textons, textonImages);
	    cout << "done" << endl;
	    // filter responses are not needed anymore
	    filterResp.clear();
	    
	    // calculate weighted texton histogram, which will serve as image descriptor
	    cout << "get texton histogram" << endl;
	    Mat textonHistogram = Mat::zeros(numberOfDBImages, numberOfTextons, CV_32FC1);
	    calcTextonHistograms(textonImages, textonHistogram);
	    cout << "done" << endl;
	    // save image descriptors
	    saveMat(textonHistogram, databaseDescrFile);

	    cout << endl << "System is now ready to look for query images" << endl;
	    cout << "Please run: dip3 find <init-file>" << endl;
	  }

	  // to image descriptor for query images and look for them within the database
	  if (strcmp(argv[1], "find") == 0){
	      
	      // load pre-calculated textons
	      Mat textons = loadMat(textonsFile);
	      // load image descriptore database (also precalculated)
	      Mat db = loadMat(databaseDescrFile);
	  
	      // load query images as gray-scale
	      cout << "load image queries" << endl;
	      vector<Mat> queries;
	      numberOfQueryImages = loadDB(queries, queryFile, numberOfQueryImages);
	      cout << "done: There are " << queries.size() << " images" << endl;
	      
	      // add some distortion to query
	      // since "our" query equals original image
	      // if you are using a "real" query delete this line
	      distortQuery(queries);
	  
	       // apply filterbank to all images to get filter responses
	      cout << "apply filterbank" << endl;
	      vector<Mat> filterResp;
	      applyFilterbank(queries, filterResp);
	      cout << "done" << endl;
	      // original images are not needed anymore
	      queries.clear();
	  
	      // calculate the distance of each pixel (ie. filter response) to each texton
	      cout << "get texton images" << endl;
	      vector<Mat> textonImages;
	      getTextonImages(filterResp, textons, textonImages);
	      cout << "done" << endl;
	      // filter responses are not needed anymore
	      filterResp.clear();
	    
	      // calculate weighted texton histogram, which will serve as image descriptor
	      cout << "get texton histogram" << endl;
	      Mat textonHistogram = Mat::zeros(numberOfQueryImages, numberOfTextons, CV_32FC1);
	      calcTextonHistograms(textonImages, textonHistogram);
	      cout << "done" << endl;
	      // texton images are not needed anymore
	      textonImages.clear();
	      
	      // find the most similar database entry to each query
	      cout << "find query" << endl;
	      findQuery(textonHistogram, db);

	  }

	  return 0;

}

// ******************************
// TO DO FUNCTIONS START HERE  **
// ******************************

// creates a specific kernel
/*
kernel	the calculated kernel
kSize	size of the kernel
name	specifies which kernel shall be computed
*/
void createKernel1D(Mat& kernel, int kSize, string name){

	assert(kSize % 2 == 1);


    // alloc memory for 1D-kernel (horizontal)
    kernel = Mat(1, kSize, CV_32FC1);
  

    // standard deviation in x direction, eg.:
    double sigma_x = kernel.cols/10.;
    
    if(name == STR_GAUSSIAN){
    	double sigma_x = kernel.cols/14.;
    	for(int i=0;i<=kSize/2;i++){
    		kernel.at<float>(0,i+kSize/2) = exp(-i*i/(2*sigma_x*sigma_x))/(sqrt(2*M_PI)*sigma_x);
    	}
    	for(int i=0;i<kSize/2;i++){
    		kernel.at<float>(0,i) = kernel.at<float>(0,kSize-1-i);
    	}
    }else if(name == STR_GAUSSIANDEVX){
    	double sigma_x = kernel.cols/8.;
    	for(int i=0,x=-kSize/2; i< kSize;i++,x++){
    		kernel.at<float>(0,i) = -(x)/(sigma_x*sigma_x)*exp(-(x)*(x)/(2*sigma_x*sigma_x))/(sqrt(2*M_PI)*sigma_x);
    	}

    }else if(name == STR_GAUSSIANDEVXX){
    	double sigma_x = kernel.cols/10.;
    	for(int i=0,x=-kSize/2; i< kSize;i++,x++){
    		kernel.at<float>(0,i) = 1/(sigma_x*sigma_x)*(x*x/(sigma_x*sigma_x)-1)*exp(-x*x/(2*sigma_x*sigma_x))/(sqrt(2*M_PI)*sigma_x);
    	}
    }else if(name == STR_MEXHAT){
    	double sigma_x = kernel.cols/20.;
    	double sigma_m = sigma_x*1.6;
    	for(int i=0,x=-kSize/2;i<kSize;i++,x++){
    		kernel.at<float>(0,i) = exp(-x*x/(2*sigma_x*sigma_x))/(sqrt(2*M_PI)*sigma_x) - exp(-x*x/(2*sigma_m*sigma_m))/(sqrt(2*M_PI)*sigma_m);
    	}
    }else{
    	throw IllegalArgumentException();
    }

}

// spatial convolution
/*
in		input image
out		output image
kernel		the convolution kernel
phi		orientation
*/
void spatialConvolution(Mat& in, Mat& out, Mat& kernel, double phi){
	assert(kernel.rows==1);
	assert(kernel.cols%2==1);


	if(in.size() != out.size()){
		out = Mat(in.rows,in.cols,CV_32FC1);
	}


	phi=M_PI*phi/180;

	for(int row=0; row<in.rows; row++){
		for(int col=0; col < in.cols;col++){
			double sumv =0;
			for(int i=0,x=kernel.cols/2;i<kernel.cols;i++,x--){
				int rx=round(col+x*cos(phi));
				int ry=round(row+x*sin(phi));

				while(rx <0){
					rx += in.cols;
				}
				while(rx >= in.cols){
					rx -= in.cols;
				}

				while(ry < 0){
					ry += in.rows;
				}
				while(ry >= in.rows){
					ry -= in.rows;
				}
				sumv += kernel.at<float>(0,i)*in.at<float>(ry,rx);
			}
			out.at<float>(row,col) = sumv;
		}
	}
}

void printKernel(Mat& kernel){
	double sumv = 0;
	for(int i=0; i < kernel.cols;i++){
		cout << kernel.at<float>(0,i) << ";";
		sumv+=kernel.at<float>(0,i);
	}
	cout << endl;
	cout << "sum:" << sumv << endl;
}

// applies filter bank to images
/*
db		images
filterResp	the obtained filter responses
*/
void applyFilterbank(vector<Mat>& db, vector<Mat>& filterResp){

    // specification of MR8-filterbanke
    const int numberOfOrientations = 6;
    const int numberOfScales = 3;
    float orientation[numberOfOrientations] = {0,30,60,90,120,150};
    // scales might be changed
    //float scale[numberOfScales] = {3,7,11};		// small, fast, but what about accuracy?
    float scale[numberOfScales] = {5,17,31};		// larger, slower,  but what about accuracy?

    Mat kernel(0,0,CV_32FC1);
    Mat derivKernel(0,0,CV_32FC1);
    Mat intermediate(0,0,CV_32FC1);
    Mat result(0,0,CV_32FC1);
    int number =0;

    for(vector<Mat>::iterator it = db.begin(); it != db.end(); it++){

    	// Gaussian filter
    	createKernel1D(kernel,scale[numberOfScales-1],STR_GAUSSIAN);
    	spatialConvolution(*it,intermediate,kernel,0);
    	spatialConvolution(intermediate,result,kernel,90);
//    	imwrite("gaussian.jpg",result);
    	filterResp.push_back(result);
    	//Laplacian Gaussian filter
    	createKernel1D(kernel,scale[numberOfScales-1],STR_MEXHAT);
    	spatialConvolution(*it,intermediate,kernel,0);
    	spatialConvolution(intermediate,result,kernel,90);
    	double maxvalue=0;
    	for(int i=0; i < result.rows;i++){
    		for(int j=0; j < result.cols; j++){
    			if(maxvalue < abs(result.at<float>(i,j))){
    				maxvalue =abs(result.at<float>(i,j));
    			}
    		}
    	}
    	double scalingFactor = 255/maxvalue;
//    	imwrite("laplace.jpg",abs(result)*scalingFactor);
    	filterResp.push_back(result);
    	// 1st derivative of Gaussian
    	for(int i=0; i < numberOfScales;i++){
    		Mat maxValue = Mat((*it).rows,it->cols,CV_32FC1,(Scalar)0);
    		createKernel1D(kernel,scale[i],STR_GAUSSIAN);
			createKernel1D(derivKernel,scale[i],STR_GAUSSIANDEVX);
    		for(int j=0; j< numberOfOrientations;j++){
    			spatialConvolution(*it,intermediate,derivKernel,orientation[j]);
    			spatialConvolution(intermediate,result,kernel,orientation[j]+90);
    			cv::max(result,maxValue,maxValue);
    		}
    		double maxvalue=0;
			for(int l=0; l < result.rows;l++){
				for(int j=0; j < result.cols; j++){
					if(maxvalue < abs(result.at<float>(l,j))){
						maxvalue =abs(result.at<float>(l,j));
					}
				}
			}
			double scalingFactor = 255/maxvalue;
			maxValue *=scalingFactor;
//    		stringstream ss;
//    		ss << "1deriv" << i << ".jpg";
//    		imwrite(ss.str(),maxValue);

    		filterResp.push_back(maxValue);
    	}

    	// 2nd derivative of Gaussian
    	for(int i=0; i < numberOfScales;i++){
			Mat max = Mat((*it).rows,it->cols,CV_32FC1,(Scalar)0);
			createKernel1D(kernel,scale[i],STR_GAUSSIAN);
			createKernel1D(derivKernel,scale[i],STR_GAUSSIANDEVXX);
			for(int j=0; j< numberOfOrientations;j++){
				spatialConvolution(*it,intermediate,derivKernel,orientation[j]);
				spatialConvolution(intermediate,result,kernel,orientation[j]+90);
				cv::max(result,max,max);
			}
			double maxvalue=0;
			for(int l=0; l < result.rows;l++){
				for(int j=0; j < result.cols; j++){
					if(maxvalue < abs(result.at<float>(l,j))){
						maxvalue =abs(result.at<float>(l,j));
					}
				}
			}
			double scalingFactor = 255/maxvalue;
			max *=scalingFactor;
//			stringstream ss;
//			ss << "2deriv" << i << ".jpg";
//			imwrite(ss.str(),max);
			filterResp.push_back(max);
		}
    	cout << "Image nr. " << ++number << " treated" << endl;
    }
}

// calculates texton image, where each pixel corresponds to distance to the particular texton
/*
filterResp	filter response images
textons		textons
textonImages	texton images
*/
void getTextonImages(vector<Mat>& filterResp, Mat& textons, vector<Mat>& textonImages){
	assert(filterResp.size()%textons.cols==0);

	for(int i=0; i < filterResp.size(); i+=textons.cols){
		for(int j=0; j< textons.rows;j++){
			Mat workingMat = Mat(filterResp[i].rows,filterResp[i].cols,CV_32FC1);
			for(int y=0; y< workingMat.rows;y++){
				for(int x=0; x < workingMat.cols;x++){
					double sumv =0;
					for(int l = 0; l < textons.cols;l++){
						sumv += (filterResp[i+l].at<float>(y,x)-textons.at<float>(j,l))*(filterResp[i+l].at<float>(y,x)-textons.at<float>(j,l));
					}
					workingMat.at<float>(y,x) = sqrt(sumv);
				}
			}
			textonImages.push_back(workingMat);
		}
	}
}

// calculates texton histograms
/*
textonImages		texton images
textonHistogram		texton histograms for each image
*/
void calcTextonHistograms(vector<Mat>& textonImages, Mat& textonHistogram){
    assert(textonImages.size() % textonHistogram.cols == 0);
    assert(textonImages.size()/textonHistogram.cols == textonHistogram.rows);
    vector<Mat>::iterator it = textonImages.begin();

    for(int i =0; i< textonHistogram.rows;i++){
    	for(int j=0; j< textonHistogram.cols;j++){
    		double sumv =0;
    		for(int y =0; y< it->rows;y++){
    			for(int x =0; x < it->cols; x++){
    				sumv += it->at<float>(y,x);
    			}
    		}
    		textonHistogram.at<float>(i,j) = sumv;
    		it++;
    	}
    	double sumv =0;
    	for(int j=0; j < textonHistogram.cols;j++){
    		sumv += textonHistogram.at<float>(i,j);
    	}
    	for(int j=0; j < textonHistogram.cols;j++){
    		textonHistogram.at<float>(i,j) /=sumv;
    	}
    }
}

// finds query image in database given texton histogram as image descriptor
/*
textonHistogram		textonHistogram of the query images
db			textonHistogram of the database images
*/
void findQuery(Mat& textonHistogram, Mat& db){
	int acceptanceRate =0;
	for(int row =0; row< textonHistogram.rows;row++){
		double minDistance = numeric_limits<double>::max();
		int minIndex =-1;
		for(int dbrow =0; dbrow < db.rows;dbrow++){
			double sumv =0;
			for(int j =0; j< db.cols;j++){
				sumv += (db.at<float>(dbrow,j)-textonHistogram.at<float>(row,j))*(db.at<float>(dbrow,j)-textonHistogram.at<float>(row,j));
			}
			sumv = sqrt(sumv);
			if(minDistance > sumv){
				minDistance = sumv;
				minIndex = dbrow;
			}
		}
		cout << "Query:" << row << " best match:" << minIndex << endl;

		if(row == minIndex){
			acceptanceRate++;
		}
	}

	cout << "Acceptance Rate:" << (float)acceptanceRate/textonHistogram.rows << endl;
}

// ******************************
// GIVEN FUNCTIONS START HERE  **
// ******************************

// load image database
/*
db	the database
fname	path to file containing all the paths
*/
int loadDB(vector<Mat>& db, string fname, int numberOfImages){
  
    // open file
    ifstream dbfile(fname.c_str());
    // if not successfull, throw error and exit
    if (!dbfile){
	cerr << "ERROR: Cannot open file " << fname << endl;
	exit(-1);
    }
    
    // read file line by line
    string buffer;
    int num = 0;
    while(getline(dbfile, buffer)){
	cout << "\t> load image from " << buffer << endl;
	// read image
	Mat img = imread(buffer,0);
	// if not successfull, throw error and exit
	if (!img.data){
	    cerr << "ERROR: file " << buffer << " not found" << endl;
	    exit(-1);
	}
	// convert to floating point precision
	img.convertTo(img,CV_32FC1);
	db.push_back(img);
	num++;
	// break if maximal amount of images is reached
	if (num == numberOfImages)
	  break;
    }
    // return number of loaded images
    return num;
}

// add distortion to images
/*
queries		vector of query images
*/
void distortQuery(vector<Mat>& queries){
    // loop through all images
    for(vector<Mat>::iterator img = queries.begin(); img != queries.end(); img++){
	// generate gaussian noise...
	Mat noise((*img).size(), CV_32FC1);
	randn(noise, Scalar::all(0), Scalar::all(10));
	// and add it to the image
	(*img) += noise;
	// constrain image values to be in [0,255]
	threshold((*img), (*img), 0, 0, THRESH_TOZERO);
	threshold((*img), (*img), 255, 255, THRESH_TRUNC);
	// finally flip image
	flip((*img), (*img), 0);
    }
}

// performs kmeans clustering
/*
filterResp		filter response images
textons			the calculated textons, ie. cluster centers
numberOfDataPoints 	number of data points used for clustering (randomly sampled)
*/
void clustering(vector<Mat>& filterResp, Mat& textons, int numberOfDataPoints){

    // the number of clusters
    int numberOfCluster = textons.rows;
    // number of images
    int numberOfImages = filterResp.size()/8;

    // the sample matrix
    Mat samples(numberOfDataPoints, 8, CV_32FC1);
    // randomly sample data points from images
    // in order to speed up the clustering
    for(int i=0; i<numberOfDataPoints; i++){
	int imgIndex = rand() % numberOfImages;
	int y = rand() % filterResp.at(imgIndex*8).rows;
	int x = rand() % filterResp.at(imgIndex*8).cols;
	for(int r=0; r<8; r++){
	    samples.at<float>(i,r) = filterResp.at(imgIndex*8+r).at<float>(y, x);
	}
    }
  
    // dummy
    Mat labels(numberOfDataPoints, 1, CV_32FC1);
    
    // perform k-means clustering
    // number of maximal iterations
    int maxIter = 1000;
    kmeans(samples, numberOfCluster, labels, TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, maxIter, 0.001), 5, KMEANS_PP_CENTERS, textons);

}

// loads a matrix given in a specific format
/*
fname	path to file
*/
Mat loadMat(string fname){

    // open file
    ifstream file(fname.c_str());
    
    // if file couldnt be opened: throw error and exit
    if (!file){
	cerr << "ERROR: Cannot open file " << fname << endl;
	exit(-1);
    }
    
    int pos;
    string buffer;
    
    // read first line containing number of rows
    getline(file, buffer);
    // set rows
    int rows = atof(buffer.c_str());
    // read second line containing number of columns
    getline(file, buffer);
    // set columns
    int cols = atof(buffer.c_str());
    
    // create matrix object, that needs to be filled
    Mat matrix = Mat(rows, cols, CV_32FC1);
    
    // read rows (one per line)
    for(int r=0; r<rows; r++){
	// erease leading "[" and tailing "]"
	getline(file, buffer);
	pos = buffer.find('[');
	if (pos >= 0) buffer.erase(pos,1);
	pos = buffer.find(']');
	if (pos >= 0) buffer.erase(pos,1);
	// get column entry of current row (divided by ",")
	for(int c=0; c<cols-1; c++){
	    pos = buffer.find_first_of(",");
	    matrix.at<float>(r,c) = atof(buffer.substr(0,pos).c_str());
	    buffer = buffer.substr(pos+2, string::npos);
	}
	// ignore the ";" at the end of the last column
	matrix.at<float>(r,cols-1) = atof(buffer.substr(0,buffer.length()-1).c_str());
    }
    // close the file
    file.close();
    
    return matrix;
}

// saves a matrix in a specific format
/*
matrix	the matrix to be saved
fname	path to file
*/
void saveMat(Mat matrix, string fname){

    // open file
    ofstream file(fname.c_str());
    // write matrix dimensions to file
    file << matrix.rows << endl;
    file << matrix.cols << endl;
    // write matrix to file
    file << matrix << endl;
    // close file
    file.close();

}

// simple function to load text-based init file
/*
fname	path of init file
*/
void readInitFile(string fname, int& numberOfTextons, string& imgDBFile, string& queryFile, string& textonsFile, string& databaseDescrFile, double& numberOfDataPoints, int& numberOfDBImages, int& numberOfQueryImages){
  
    // open file
    ifstream file(fname.c_str());
  
    // if file doesnt exist, return
    if (!file){
      cerr << "ERROR: Cannot open file " << fname << endl;
      exit(-1);
    }
  
    // else: read file
    // line per line, save each line in buffer
    string buffer;
    while(  getline ( file, buffer ) ){

	// if line is a comment (or empty) continue
	if (isCommentline(&buffer))
	    continue;
    
	// set number of textons
	if (buffer.compare("numberOfTextons")==0){
	    getline ( file, buffer );
	    numberOfTextons = atoi(buffer.c_str());
	    continue;
	}
	// set file to load image database
	if (buffer.compare("imgDBFile")==0){
	    getline ( file, buffer );
	    imgDBFile = buffer;
	    continue;
	}
	// set file to load query images
	if (buffer.compare("queryFile")==0){
	    getline ( file, buffer );
	    queryFile = buffer;
	    continue;
	}
	// set file to save/load textons
	if (buffer.compare("textonsFile")==0){
	    getline ( file, buffer );
	    textonsFile = buffer;
	    continue;
	}
	// set file to save/load database image descriptors
	if (buffer.compare("databaseDescrFile")==0){
	    getline ( file, buffer );
	    databaseDescrFile = buffer;    
	    continue;
	}
	// set number of datapoints used during clustering
	if (buffer.compare("numberOfDataPoints")==0){
	    getline ( file, buffer );
	    numberOfDataPoints = atof(buffer.c_str());    
	    continue;
	}
	// set maximal number of images loaded from image database
	if (buffer.compare("numberOfDBImages")==0){
	    getline ( file, buffer );
	    numberOfDBImages = atoi(buffer.c_str());
	    continue;
	}
	// set maximal number of query images
	if (buffer.compare("numberOfQueryImages")==0){
	    getline ( file, buffer );
	    numberOfQueryImages = atoi(buffer.c_str());
	    continue;
	}
    }
}

// check for comment lines within the init file
bool isCommentline(string* buffer){

    // if line is empty or starts with # (which means its a comments)
    if ((buffer->length()==0) or (buffer->at(0) == '#'))
      return true;
  
    // delete leading blanks
    if (buffer->at(0)== ' '){
      buffer->erase(0, buffer->find_first_not_of(" "));
    }

    // delete blanks at end of string
    int diff = (buffer->length() - buffer->find_last_not_of(" ") - 1);
    if (diff !=0)
      buffer->erase(buffer->find_last_not_of(" ") + 1, diff);
  
    // if line is empty or starts with # (which means its a comments)
    if ((buffer->length()==0) or (buffer->at(0) == '#'))
      return true;

    return false;
}
