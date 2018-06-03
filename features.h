/*
 * features.h
 *
 *  Contains audio feature extraction methods.
 */

#ifndef FEATURES_H_
#define FEATURES_H_

/**
 * Computes the spectral centroid, spread, skewness, and kurtosis
 */
double *spectralShapeStatistics(double *mag);

/*
 * Computes the nth moment
 */
double moment(double* mag, int n);


#endif /* FEATURES_H_ */
