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

float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

float predict_probability(const float *x, const float *weights, const float bias) {
    float z = bias;
    for (int j = 0; j < FEATURE_COUNTS; j++) {
        z += x[j] * weights[j];
    }
    return sigmoid(z);
}

int predict_class(const float *x, const float *weights, const float bias) {
    return predict_probability(x, weights, bias) >= THRESHOLD ? 1 : 0;
}

float calculate_cost(const float *x, const int *y, const float *weights, const float bias) {
    float cost = 0.0f;
    for (int i = 0; i < DATA_POINTS; i++) {
        float h = predict_probability(&IDX(x, i, 0), weights, bias);
        cost += -y[i] * logf(h) - (1 - y[i]) * logf(1 - h);
    }
    return cost / DATA_POINTS;
}

float* calculate_weights_gradients(const float *x, const int *y, const float *weights, const float bias) {
    float *gradients = (float*)calloc(FEATURE_COUNTS, sizeof(float));

    for (int i = 0; i < DATA_POINTS; i++) {
        float h = predict_probability(&IDX(x, i, 0), weights, bias);
        float error = h - y[i];

        for (int j = 0; j < FEATURE_COUNTS; j++) {
            gradients[j] += error * IDX(x, i, j);
        }
    }

    for (int j = 0; j < FEATURE_COUNTS; j++) {
        gradients[j] /= DATA_POINTS;
    }

    return gradients;
}

float calculate_bias_gradient(const float *x, const int *y, const float *weights, const float bias) {
    float gradient = 0.0f;

    for (int i = 0; i < DATA_POINTS; i++) {
        float h = predict_probability(&IDX(x, i, 0), weights, bias);
        gradient += h - y[i];
    }

    return gradient / DATA_POINTS;
}

float* run_gradient_descent(float *x, int *y, float alpha) {
    float *weights = (float*)calloc(FEATURE_COUNTS, sizeof(float));
    float bias = 0.0f;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float *weight_gradients = calculate_weights_gradients(x, y, weights, bias);
        float bias_gradient = calculate_bias_gradient(x, y, weights, bias);

        for (int j = 0; j < FEATURE_COUNTS; j++) {
            weights[j] -= alpha * weight_gradients[j];
        }
        bias -= alpha * bias_gradient;

        free(weight_gradients);

        if (epoch % 100 == 0) {
            float cost = calculate_cost(x, y, weights, bias);
            printf("Epoch %d, Cost: %.6f\n", epoch, cost);
        }
    }

    float *result = (float*)malloc((FEATURE_COUNTS + 1) * sizeof(float));
    for (int j = 0; j < FEATURE_COUNTS; j++) {
        result[j] = weights[j];
    }
    result[FEATURE_COUNTS] = bias;

    free(weights);
    return result;
}

int main() {
    float x[DATA_POINTS * FEATURE_COUNTS];
    int y[DATA_POINTS];

    srand(42);

    for (int i = 0; i < DATA_POINTS; i++) {
        IDX(x, i, 0) = (float)rand() / RAND_MAX * 10 - 5;
        IDX(x, i, 1) = (float)rand() / RAND_MAX * 10 - 5;

        float decision_boundary = IDX(x, i, 0) + IDX(x, i, 1) - 1;
        y[i] = decision_boundary > 0 ? 1 : 0;

        if (rand() % 10 == 0) {
            y[i] = 1 - y[i];
        }
    }

    printf("Training logistic regression...\n");
    float *model = run_gradient_descent(x, y, LEARNING_RATE);

    printf("\nFinal weights: [%.4f, %.4f], bias: %.4f\n",
           model[0], model[1], model[FEATURE_COUNTS]);

    int correct = 0;
    for (int i = 0; i < DATA_POINTS; i++) {
        int predicted = predict_class(&IDX(x, i, 0), model, model[FEATURE_COUNTS]);
        if (predicted == y[i]) correct++;
    }

    printf("Training accuracy: %.2f%%\n", (float)correct / DATA_POINTS * 100);

    free(model);
    return 0;
}
