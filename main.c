#include "DSP_Config.h"
#include <stdio.h>
#include "fft.h"
#include "gmm.h"
#include "libmfcc.h"
#include <math.h>
#include "features.h"

#define BUFFERSIZE 1024
int M=BUFFERSIZE;
int kk=0;
int startflag = 0;
int training = 1;

short X[BUFFERSIZE];
COMPLEX w[BUFFERSIZE];
COMPLEX B[BUFFERSIZE];
double spectralData[BUFFERSIZE];
float magnitude = 0;
float avg = 0;

int coeff=0;
double spectrum[BUFFERSIZE];
double mfcc[13];
double *sss;
double feats[17];
float llh;

double weights[3] = {0.33363005,0.49528663,0.17108333};
double means[51] = {2.29714624,-0.46984484,0.47660914,-0.32027861,-0.17388577,-0.21813227,-0.04568859,-0.09759609,-0.08454685,-0.03303321,0.03034760,-0.02713213,0.07924577,56.09046496,71.17555198,2.82405274,3.00807476,3.62933477,-0.33285165,0.69907121,-0.09367394,-0.06603198,-0.18683111,-0.02273308,-0.07654318,-0.10898050,-0.12647405,-0.07157140,-0.11925906,-0.03874968,35.61909844,54.21336795,4.09823706,18.37054612,4.85423335,-0.40543869,0.62351327,-0.23289740,-0.11255583,-0.20642021,-0.16982797,-0.14830839,-0.14771997,-0.10197164,-0.04251470,-0.09897561,0.00543935,23.70044223,37.79146474,6.28980612,52.57300572};
double covars[51] = {0.84875131,0.54540867,0.32973933,0.20418287,0.24063123,0.22618938,0.21946397,0.26370630,0.26671025,0.31496457,0.31107896,0.30793353,0.27373178,172.22349796,306.23727274,0.48076357,57.22554751,0.49993913,0.29117162,0.19743981,0.16691678,0.12141554,0.11483228,0.12981453,0.11895683,0.12166940,0.13401627,0.12694267,0.13109478,0.13461637,57.06804518,84.98613898,0.71991817,94.23336426,0.64083104,0.58378049,0.24803646,0.18978901,0.14482897,0.10113280,0.16910565,0.13068864,0.14154063,0.12612072,0.12279887,0.14034926,0.16337058,30.47268890,44.44756892,2.18066400,854.49129325};
GMM gmm[1]; // create GMM model

int main()
{
	int K = 3; // Number of Classes
	int D = 17; // Number of Features

	DSP_Init();

	int ii, mm, ll;

	// Initialize GMM model
	gmm_new(gmm, K, D, "diagonal");
	gmm_set_convergence_tol(gmm, 1e-6);
	gmm_set_regularization_value(gmm, 1e-6);
	gmm_set_initialization_method(gmm, "random");

	for (ii=0; ii < 3; ii++) {
		gmm->weights[ii] = weights[ii];
	}
	for (ii=0; ii < 51; ii++) {
		gmm->means[ii] = means[ii];
	}
	for (ii=0; ii < 51; ii++) {
		gmm->covars[ii] = covars[ii];
	}

	// Twiddle factor
	for(ii=0; ii < BUFFERSIZE; ii++) {
		w[ii].real = cos((float)ii/(float)BUFFERSIZE*PI);
		w[ii].imag = sin((float)ii/(float)BUFFERSIZE*PI);
	}

	// main stalls here, interrupts drive operation
	while(1) {
		if(startflag){
			// Remove bias (DC offset)
			avg = 0;
			for(mm=0; mm < BUFFERSIZE; mm++){
				avg = avg + X[mm];
			}
			// Measure the Magnitude of the input to find starting point
			avg = avg/BUFFERSIZE;
			magnitude = 0;
			for(mm=0; mm < BUFFERSIZE; mm++){
				magnitude = magnitude + abs(X[mm]-avg);
			}

			if(magnitude > 30000) {

				for(ll=0; ll<M; ll++){
					B[ll].real = X[ll]-avg;
					B[ll].imag = 0;
				}
				// (P3). FFT: B is input and output, w is twiddle factors
				fft(B, BUFFERSIZE, w);

				for (ii = 0; ii < BUFFERSIZE; ii++) {
					spectralData[ii] = sqrt(B[ii].real*B[ii].real+B[ii].imag*B[ii].imag);
				}

				// (P3). Find 13 MFCC coefficients
				for (ii = 0; ii < 17; ii++) {
					mfcc[ii] = GetCoefficient(spectralData, 12000, 48, BUFFERSIZE, ii);
				}

				// Get the spectral shape statistics
				sss = spectralShapeStatistics(spectralData);

				// Combine features into a single feature vector
				for (ii = 0; ii < 17; ii++) {
					if (ii < 13) {
						feats[ii] = mfcc[ii];
					} else {
						feats[ii] = sss[ii];
					}
				}

				llh = gmm_score(gmm, feats, 1);
			}
			startflag = 0;
		}
	}
}
