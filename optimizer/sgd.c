#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define MAX_ITERATIONS (1000)
#define LEARNING_RATE (0.01f)
#define TOLERANCE (1e-6f)
#define BATCH_SIZE (32)

typedef struct {
    float *features;
    float target;
} DataPoint;

typedef struct {
    DataPoint *data;
    int size;
    int feature_count;
} Dataset;

typedef struct {
    float *weights;
    float bias;
    int feature_count;
} LinearModel;

float predict(LinearModel *model, float *features) {
    float result = model->bias;
    for (int i = 0; i < model->feature_count; i++) {
        result += model->weights[i] * features[i];
    }
    return result;
}

float compute_loss(LinearModel *model, Dataset *dataset) {
    float total_loss = 0.0f;
    for (int i = 0; i < dataset->size; i++) {
        float prediction = predict(model, dataset->data[i].features);
        float error = prediction - dataset->data[i].target;
        total_loss += error * error;
    }
    return total_loss / (2.0f * dataset->size);
}

void sgd_step(LinearModel *model, DataPoint *point, float learning_rate) {
    float prediction = predict(model, point->features);
    float error = prediction - point->target;
    
    model->bias -= learning_rate * error;
    for (int i = 0; i < model->feature_count; i++) {
        model->weights[i] -= learning_rate * error * point->features[i];
    }
}

void shuffle_dataset(Dataset *dataset) {
    for (int i = dataset->size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        DataPoint temp = dataset->data[i];
        dataset->data[i] = dataset->data[j];
        dataset->data[j] = temp;
    }
}

void sgd_train(LinearModel *model, Dataset *dataset, float learning_rate, int max_iterations) {
    float prev_loss = INFINITY;
    
    for (int epoch = 0; epoch < max_iterations; epoch++) {
        shuffle_dataset(dataset);
        
        for (int i = 0; i < dataset->size; i++) {
            sgd_step(model, &dataset->data[i], learning_rate);
        }
        
        float current_loss = compute_loss(model, dataset);
        printf("Epoch %d: Loss = %.6f\n", epoch, current_loss);
        
        if (fabs(prev_loss - current_loss) < TOLERANCE) {
            printf("Converged at epoch %d\n", epoch);
            break;
        }
        
        prev_loss = current_loss;
    }
}

Dataset* create_synthetic_dataset(int size, int feature_count) {
    Dataset *dataset = malloc(sizeof(Dataset));
    dataset->size = size;
    dataset->feature_count = feature_count;
    dataset->data = malloc(size * sizeof(DataPoint));
    
    float true_weights[] = {2.5f, -1.3f, 0.8f};
    float true_bias = 1.2f;
    
    for (int i = 0; i < size; i++) {
        dataset->data[i].features = malloc(feature_count * sizeof(float));
        dataset->data[i].target = true_bias;
        
        for (int j = 0; j < feature_count; j++) {
            dataset->data[i].features[j] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
            dataset->data[i].target += true_weights[j] * dataset->data[i].features[j];
        }
        
        dataset->data[i].target += ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
    
    return dataset;
}

LinearModel* create_model(int feature_count) {
    LinearModel *model = malloc(sizeof(LinearModel));
    model->feature_count = feature_count;
    model->weights = malloc(feature_count * sizeof(float));
    model->bias = 0.0f;
    
    for (int i = 0; i < feature_count; i++) {
        model->weights[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
    
    return model;
}

void free_dataset(Dataset *dataset) {
    for (int i = 0; i < dataset->size; i++) {
        free(dataset->data[i].features);
    }
    free(dataset->data);
    free(dataset);
}

void free_model(LinearModel *model) {
    free(model->weights);
    free(model);
}

void test_sgd() {
    srand(42);
    
    printf("Testing SGD on synthetic linear regression problem\n");
    printf("True model: y = 2.5*x1 - 1.3*x2 + 0.8*x3 + 1.2 + noise\n\n");
    
    Dataset *dataset = create_synthetic_dataset(1000, 3);
    LinearModel *model = create_model(3);
    
    printf("Initial loss: %.6f\n", compute_loss(model, dataset));
    printf("Initial weights: [%.3f, %.3f, %.3f], bias: %.3f\n\n", 
           model->weights[0], model->weights[1], model->weights[2], model->bias);
    
    sgd_train(model, dataset, LEARNING_RATE, 100);
    
    printf("\nFinal loss: %.6f\n", compute_loss(model, dataset));
    printf("Final weights: [%.3f, %.3f, %.3f], bias: %.3f\n", 
           model->weights[0], model->weights[1], model->weights[2], model->bias);
    printf("Expected weights: [2.500, -1.300, 0.800], bias: 1.200\n");
    
    assert(fabs(model->weights[0] - 2.5f) < 0.1f);
    assert(fabs(model->weights[1] - (-1.3f)) < 0.1f);
    assert(fabs(model->weights[2] - 0.8f) < 0.1f);
    assert(fabs(model->bias - 1.2f) < 0.1f);
    
    printf("\nTest passed: SGD converged to expected parameters\n");
    
    free_dataset(dataset);
    free_model(model);
}

int main() {
    test_sgd();
    return 0;
}