#include "DSP_Config.h"
#include <stdio.h>
#include "fft.h"
#include "gmm.h"
#include "libmfcc.h"
#include <math.h>

#define BUFFERSIZE 512
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
double feats[17];
float llh;

double weights[3] = {0.13691338,0.42999811,0.43308851};
double means[51] = {8.41805802,-1.20110115,0.92925516,-0.24406504,-0.11253630,-0.19882333,0.14178768,0.03575083,-0.07376402,-0.36426823,-0.28853406,-0.07004035,0.08715983,249.11887183,226.88127495,0.06045491,-18.20852378,8.23594474,-1.62989906,0.55335641,-0.40107086,-0.53878990,-0.57618524,-0.11515633,-0.20056607,-0.19102873,-0.44693833,-0.37105288,-0.19977371,0.02434016,255.17447553,226.61086489,0.01389267,-19.11809376,11.06796020,-1.13093843,0.86927381,-0.14007003,-0.43669937,-0.52863677,-0.13197519,-0.24376373,-0.21498049,-0.49623460,-0.48275173,-0.26451042,-0.04108370,254.04098703,236.55539473,0.02287205,-16.81242407};
double covars[51] = {12.32616374,4.41928576,1.58055229,1.17943921,0.96403003,0.96906832,0.81789760,0.73581008,0.73146231,0.69604027,0.68741466,0.70048999,0.58218482,66.80846707,128.50600365,0.00407604,11.07518030,1.33407821,2.86550499,1.33047960,0.95484396,0.81002713,0.73158922,0.66157717,0.63064384,0.63266493,0.53804487,0.56659392,0.58654053,0.56533020,2.41143744,10.72531248,0.00014009,0.77370140,1.80675971,3.24741240,1.35054203,0.90542282,0.81169333,0.59998876,0.52735615,0.61703685,0.54045908,0.45539269,0.47662513,0.47521282,0.46313234,5.38466110,12.18236673,0.00032190,0.59336996};
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
			for(mm = 0; mm < BUFFERSIZE; mm++){
				avg += X[mm];
			}

			// Measure the magnitude of the input to find starting point
			avg = avg/BUFFERSIZE;
			magnitude = 0;
			for(mm=0; mm < BUFFERSIZE; mm++){
				magnitude = magnitude + abs(X[mm]-avg);
			}

			if(magnitude > 30000) {

				// Eliminate bias
				for(ll = 0; ll < BUFFERSIZE; ll++){
					B[ll].real = X[ll] - avg;
					B[ll].imag = 0;
				}

				// Get the short-time fourier transform
				fft(B, BUFFERSIZE, w);

				// Get the magnitude of the FFT
				for (ii = 0; ii < BUFFERSIZE; ii++) {
					spectralData[ii] = sqrt(B[ii].real*B[ii].real + B[ii].imag*B[ii].imag);
				}

				// Get the MFCC
				for (ii = 0; ii < 13; ii++) {
					feats[ii] = GetCoefficient(spectralData, 48000, 48, BUFFERSIZE, ii);
				}

				// Get the spectral shape statistics
				double dataSum = 0.0;
				double mu[4] = {0.0, 0.0, 0.0, 0.0};

				// Get the first four raw moments
				int k = 0;
				for (k = 0; k < 512; k++) {
					double v = abs(spectralData[k]);
					dataSum += v;
					v *= k;
					mu[0] += v;
					v *= k;
					mu[1] += v;
					v *= k;
					mu[2] += v;
					v *= k;
					mu[3] += v;
				}
				mu[0] = mu[0]/dataSum; // 1st momment
				mu[1] = mu[1]/dataSum; // 2nd momment
				mu[2] = mu[2]/dataSum; // 3rd momment
				mu[3] = mu[3]/dataSum; // 4th momment

				feats[13] = mu[0]; // centroid
				feats[14] = sqrt(mu[1] - pow(mu[0], 2)); // spread
				feats[15] = (2 * pow(mu[0], 3) - 3 * mu[0] * mu[1] + mu[2])/pow(feats[14], 3); // skewness
				feats[16] = (-3 * pow(mu[0], 4) - 4 * mu[0] * mu[2] + mu[3])/pow(feats[14], 4) - 3; // kurtosis

				// Display features
				printf("Features: ");
				for (ii = 0; ii < 17; ii++) {
					printf("%f ", feats[ii]);
				}
				printf("\n");

				llh = gmm_score(gmm, feats, 1);
			}
			startflag = 0;
		}
	}
}
