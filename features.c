/*
 * features.h
 *
 *  Contains audio feature extraction methods.
 */

#include <math.h>
#include "features.h"

#define N 1024;

/**
 * Computes the spectral centroid, spread, skewness, and kurtosis
 */
double *spectralShapeStatistics(double *mag)
{
	double stats[4];
	int k;

	// Get the first four raw moments
	double mu[4];
	for (k = 0; k < 4; k++) {
		mu[k] = moment(mag, k + 1);
	}

	// Centroid
	stats[0] = mu[0];

	// Spread
	stats[1] = sqrt(mu[1] - pow(mu[0], 2));

	// Skewness
	stats[2] = (2*pow(mu[0], 3) - 3*mu[0]*mu[1] + mu[2])/(pow(stats[1], 3));

	// Kurtosis or spectral flatness
	stats[3] = (-3*pow(mu[0], 4) + 6*mu[0]*mu[1] - 4*mu[0]*mu[2] + mu[3])/(pow(stats[2], 4)) - 3;

	return stats;
}

/*
 * Computes the nth raw moment of the magnitude of the spectrum
 */
double moment(double* mag, int n)
{
	double num, den;

	int k;
	for (k = 0; k < N; k++) {
		num += pow(k, n) * mag[k];
		den += mag[k];
	}

	return num/den;
}
