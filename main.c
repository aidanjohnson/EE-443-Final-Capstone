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
double sss[4];
double feats[17];
float llh;

double weights[3] = {0.18265455,0.36151545,0.45582999};
double means[51] = {7.99306564,-2.66702059,-0.14877771,-1.00729928,-1.02711138,-0.99235752,-0.42483802,-0.45255147,-0.44734024,-0.68264314,-0.57969104,-0.35151826,-0.05523575,1.41419278,26.15998507,21.16914388,530.01064269,10.22199728,-0.48712182,1.32295514,0.23260397,0.04280672,-0.11080689,0.22418082,0.10415348,0.07511782,-0.25841487,-0.24341097,-0.06786570,0.07683761,20.38074130,95.68356965,5.30047136,27.84591123,9.50356950,-1.51778351,0.63740513,-0.36557562,-0.57934880,-0.61998910,-0.19899423,-0.31132668,-0.28693674,-0.52401222,-0.47002261,-0.26612398,-0.02869980,5.02158946,49.64112728,10.46072146,111.31037329};
double covars[51] = {5.80985953,2.75584264,1.29991573,0.92305431,0.80314060,0.74409226,0.73704814,0.77592255,0.72105652,0.60392618,0.61592461,0.69576200,0.67593837,0.33580073,37.38614191,83.88288650,3595144.46585853,3.89782447,2.59043282,1.05381713,0.68500870,0.66376731,0.59263296,0.50173277,0.50840194,0.49311974,0.45344310,0.48578981,0.48693036,0.44304774,180.55295670,683.66647300,2.00824405,212.07152715,4.12934739,2.76961048,1.11924008,0.80723114,0.67563516,0.58491213,0.56085110,0.59523182,0.56296153,0.49564159,0.53546170,0.53183538,0.52366893,3.15143528,78.44448915,3.93189659,1856.57314796};
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
				double den, num;
				double mu[4];

				// Get the first four raw moments
				int n, k;
				for (n = 1; n <= 4; n++) {
					num = 0.0;
					den = 0.0;
					for (k = 0; k < 512; k++) {
						num += pow(k, n) * spectralData[k];
						den += spectralData[k];
					}
					mu[n] = num/den;
				}

				// Get the first four moments
				feats[13] = mu[0];
				feats[14] = sqrt(mu[1] - mu[0]*mu[0]);
				feats[15] = (2*mu[0]*mu[0]*mu[0] - 3*mu[0]*mu[1] + mu[2])/(sss[1]*sss[1]*sss[1]);
				feats[16] = (-3*mu[0]*mu[0]*mu[0]*mu[0] + 6*mu[0]*mu[1] - 4*mu[0]*mu[2] + mu[3])/(sss[1]*sss[1]*sss[1]*sss[1]) - 3;

				// Display features
				printf("Features: ");
				for (ii = 0; ii < 17; ii++) {
					printf("%.4f ", feats[ii]);
				}
				printf("\n");

				llh = gmm_score(gmm, feats, 1);
			}
			startflag = 0;
		}
	}
}


