
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define EPOCHS (1000)
#define FEATURE_COUNTS (1)
#define LEARNING_RATE (0.0001f)
#define DATA_POINTS (100)

#define SLOPE (2)
#define Y_INTERCEPT (1)

#define IDX(arr, i, j) (arr[(i) * FEATURE_COUNTS + (j)])

float* run_gradient_descent(int *x, int *y, float alpha);
float* calculate_weights_gradients(const int *x, const int *y, const float *weights, const float bias);
float calculate_bias_gradient(const int *x, const int *y, const float *weights, const float bias);
float predict(const int *x, const float *weights, const float bias);

int main() {
    int* x = calloc(DATA_POINTS * FEATURE_COUNTS, sizeof(int));
    for (int i = 0; i < DATA_POINTS; i++) {
        for (int j = 0; j < FEATURE_COUNTS; j++) {  // Fixed: FEATURE_COUNTS not DATA_POINTS
            IDX(x, i, j) = i;
        }
    }

    int* y = calloc(DATA_POINTS, sizeof(int));
    for (int i = 0; i < DATA_POINTS; i++) {
        y[i] = SLOPE * i + Y_INTERCEPT;
    }

    float *params = run_gradient_descent(x, y, LEARNING_RATE);

    for (int i = 0; i < FEATURE_COUNTS; i++) {
        printf("Weight %d: %f\n", i, params[i]);
    }
    printf("Bias: %f\n", params[FEATURE_COUNTS]);  // Fixed: Valid index

    free(x);
    free(y);
    free(params);
    return 0;
}

float* run_gradient_descent(int *x, int*y, float alpha) {
    float* weights = calloc(FEATURE_COUNTS, sizeof(float));
    float bias = 0;
    assert(weights != NULL);

    for (int i = 0; i < EPOCHS; i++) {
        float* weights_gradient = calculate_weights_gradients(x, y, weights, bias);
        float bias_gradient = calculate_bias_gradient(x, y, weights, bias);

        for (int j = 0; j < FEATURE_COUNTS; j++) {
            weights[j] = weights[j] - alpha * weights_gradient[j];
        }
        bias = bias - alpha * bias_gradient;
        free(weights_gradient);  // Added: Prevent memory leak
    }

    float* params = calloc(FEATURE_COUNTS + 1, sizeof(float));
    for (int i = 0; i < FEATURE_COUNTS; i++) {
        params[i] = weights[i];
    }
    params[FEATURE_COUNTS] = bias;

    free(weights);
    return params;
}

float* calculate_weights_gradients(const int *x, const int *y, const float *weights, const float bias) {
    float *weight_gradient = calloc(FEATURE_COUNTS, sizeof(float));
    assert(weight_gradient != NULL);

    for (int i = 0; i < DATA_POINTS; i++) {
        float prediction = predict(&IDX(x,i,0), weights, bias);
        for (int j = 0; j < FEATURE_COUNTS; j++) {
            weight_gradient[j] += (prediction - y[i]) * IDX(x, i, j);
        }
    }

    for (int j = 0; j < FEATURE_COUNTS; j++) {
        weight_gradient[j] = weight_gradient[j] / DATA_POINTS;
    }

    return weight_gradient;
}

float calculate_bias_gradient(const int *x, const int *y, const float *weights, const float bias) {
    float bias_gradients = 0;
    for (int i = 0; i < DATA_POINTS; i++) {
        float prediction = predict(&IDX(x, i, 0), weights, bias);
        bias_gradients += (prediction - y[i]);
    }
    return bias_gradients / DATA_POINTS;
}

float predict(const int *x, const float *weights, const float bias) {
    float dot = 0.0;
    for (int i = 0; i < FEATURE_COUNTS; i++) {
        dot += x[i] * weights[i];
    }
    return dot + bias;
}
