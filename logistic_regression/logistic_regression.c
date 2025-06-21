#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define EPOCHS (1000)
#define LEARNING_RATE (0.01f)
#define DATA_POINTS (100)
#define FEATURE_COUNTS (2)

#define THRESHOLD (0.5f)
#define IDX(arr, i, j) (arr[(i) * FEATURE_COUNTS + (j)])

float sigmoid(float z);
float* run_gradient_descent(float *x, int *y, float alpha);
float* calculate_weights_gradients(const float *x, const int *y, const float *weights, const float bias);
float calculate_bias_gradient(const float *x, const int *y, const float *weights, const float bias);
float predict_probability(const float *x, const float *weights, const float bias);
int predict_class(const float *x, const float *weights, const float bias);
float calculate_cost(const float *x, const int *y, const float *weights, const float bias);
