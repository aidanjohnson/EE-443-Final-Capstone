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
double feats[13];
float llh;

double weights[3] = {0.47719201,0.39606008,0.12674791};
double means[39] = {9.02845919,-2.28606379,0.93260140,-0.44478900,-0.39408150,-0.45678549,0.01363623,-0.11176037,-0.09200932,-0.40065146,-0.34495591,-0.13476062,0.05900034,10.00510639,-0.90603721,0.61535841,0.00122473,-0.48989515,-0.54030551,-0.15750755,-0.23712568,-0.24217167,-0.50151378,-0.46486293,-0.25907806,-0.02333180,9.59747592,0.74667722,0.41731129,-0.43214525,-0.42711067,-0.56773180,-0.24762503,-0.31300213,-0.35918707,-0.52980812,-0.46869873,-0.34028991,-0.11287762};
double covars[39] = {6.46748778,2.47319813,1.74208847,1.06805890,1.00422257,0.93840253,0.80289464,0.75572601,0.76741766,0.68372677,0.72325860,0.72573652,0.66355310,3.66793856,2.26143085,1.03568613,0.83767323,0.73513353,0.53089395,0.46941680,0.55824827,0.46285559,0.40226798,0.41896392,0.42807340,0.42240167,1.64520084,1.35633666,0.96230678,0.71398716,0.61807225,0.49072618,0.42690536,0.47324116,0.39106340,0.29353792,0.27156393,0.28620101,0.29959161};
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
	for (ii=0; ii < 39; ii++) {
		gmm->means[ii] = means[ii];
	}
	for (ii=0; ii < 39; ii++) {
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

				// Display features
				printf("Features: ");
				for (ii = 0; ii < 13; ii++) {
					printf("%f ", feats[ii]);
				}
				printf("\n");

				llh = gmm_score(gmm, feats, 1);
			}
			startflag = 0;
		}
	}
}
