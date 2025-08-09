#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define MAX_ITERATIONS (1000)
#define LEARNING_RATE (0.01f)
#define TOLERANCE (1e-6f)

typedef float (*ObjectiveFunction)(float x);
typedef float (*GradientFunction)(float x);

float gradient_descent_1d(ObjectiveFunction f, GradientFunction df, float initial_x);
float quadratic_function(float x);
float quadratic_gradient(float x);

float gradient_descent_1d(ObjectiveFunction f, GradientFunction df, float initial_x) {
    float x = initial_x;

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        printf("Iteration %d: x = %.6f, f(x) = %.6f\n", i, x, f(x));
        float gradient = df(x);
        float new_x = x - LEARNING_RATE * gradient;

        if (fabs(new_x - x) < TOLERANCE) {
            return new_x;
        }

        x = new_x;
    }

    return x;
}

float quadratic_function(float x) {
    return (x - 3) * (x - 3) + 2;
}

float quadratic_gradient(float x) {
    return 2 * (x - 3);
}


int main() {
    printf("Minimizing f(x) = (x-3)^2 + 2\n");
    printf("Expected minimum: x = 3, f(x) = 2\n\n");

    float result = gradient_descent_1d(quadratic_function, quadratic_gradient, -2.0f);
    printf("Final result: x = %.6f, f(x) = %.6f\n", result, quadratic_function(result));

    return 0;
}
