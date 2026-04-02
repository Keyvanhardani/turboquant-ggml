/*
 * calibrate_codebook.c — Measure actual distribution and compute optimal
 * Lloyd-Max codebooks via iterative convergence.
 *
 * Generates random unit vectors in R^32, applies WHT, measures the
 * distribution, and runs Lloyd-Max iteration to find optimal centroids.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N 32
#define N_SAMPLES 1000000

static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static void wht32(float *x) {
    for (int len = 1; len < N; len <<= 1) {
        for (int i = 0; i < N; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = x[i+j], v = x[i+j+len];
                x[i+j] = u + v;
                x[i+j+len] = u - v;
            }
        }
    }
    float s = 1.0f / sqrtf((float)N);
    for (int i = 0; i < N; i++) x[i] *= s;
}

static float vec_norm(const float *x, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i]*x[i];
    return sqrtf(s);
}

/* Collect samples from the actual distribution */
static float *collect_samples(int n_samples) {
    float *samples = malloc((size_t)n_samples * N * sizeof(float));
    float vec[N];
    int idx = 0;

    for (int s = 0; s < n_samples; s++) {
        /* Random Gaussian vector */
        for (int i = 0; i < N; i++) vec[i] = randn();

        /* Normalize to unit sphere */
        float norm = vec_norm(vec, N);
        for (int i = 0; i < N; i++) vec[i] /= norm;

        /* WHT */
        wht32(vec);

        /* Store all N coordinates */
        for (int i = 0; i < N; i++) {
            samples[idx++] = vec[i];
        }
    }
    return samples;
}

/* Lloyd-Max iteration */
static void lloyd_max(const float *samples, int n_samples,
                      float *centroids, float *boundaries,
                      int n_levels, int max_iter) {
    int total = n_samples * N;

    /* Initialize centroids: uniform spacing in [-0.4, 0.4] */
    for (int i = 0; i < n_levels; i++) {
        centroids[i] = -0.4f + 0.8f * (float)i / (float)(n_levels - 1);
    }

    float prev_mse = 1e30f;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Update boundaries: midpoints of adjacent centroids */
        for (int i = 0; i < n_levels - 1; i++) {
            boundaries[i] = (centroids[i] + centroids[i+1]) / 2.0f;
        }

        /* Assign samples to nearest centroid and compute new centroids */
        double sums[64] = {0};
        int counts[64] = {0};

        for (int s = 0; s < total; s++) {
            float val = samples[s];
            /* Find bin via boundaries */
            int bin = 0;
            for (int b = 0; b < n_levels - 1; b++) {
                if (val >= boundaries[b]) bin = b + 1;
            }
            sums[bin] += val;
            counts[bin]++;
        }

        /* Update centroids */
        double mse = 0;
        for (int i = 0; i < n_levels; i++) {
            if (counts[i] > 0) {
                centroids[i] = (float)(sums[i] / counts[i]);
            }
        }

        /* Compute MSE */
        for (int s = 0; s < total; s++) {
            float val = samples[s];
            int bin = 0;
            for (int b = 0; b < n_levels - 1; b++) {
                if (val >= boundaries[b]) bin = b + 1;
            }
            float err = val - centroids[bin];
            mse += err * err;
        }
        mse /= total;

        if (iter % 20 == 0 || iter == max_iter - 1) {
            printf("  iter %3d: MSE = %.8f\n", iter, mse);
        }

        if (fabsf((float)(mse - prev_mse)) < 1e-12f) {
            printf("  Converged at iter %d\n", iter);
            break;
        }
        prev_mse = mse;
    }

    /* Final boundaries */
    for (int i = 0; i < n_levels - 1; i++) {
        boundaries[i] = (centroids[i] + centroids[i+1]) / 2.0f;
    }
}

static void print_array(const char *name, const float *arr, int n) {
    printf("static const float %s[%d] = {\n    ", name, n);
    for (int i = 0; i < n; i++) {
        printf("%.7ff", arr[i]);
        if (i < n-1) printf(", ");
        if ((i+1) % 4 == 0 && i < n-1) printf("\n    ");
    }
    printf("\n};\n");
}

int main(void) {
    srand(42);

    printf("Collecting %d samples (unit sphere R^32, WHT-rotated)...\n", N_SAMPLES);
    float *samples = collect_samples(N_SAMPLES);

    /* Distribution statistics */
    double sum = 0, sum2 = 0;
    float vmin = samples[0], vmax = samples[0];
    int total = N_SAMPLES * N;
    for (int i = 0; i < total; i++) {
        sum += samples[i];
        sum2 += samples[i] * samples[i];
        if (samples[i] < vmin) vmin = samples[i];
        if (samples[i] > vmax) vmax = samples[i];
    }
    printf("Distribution: mean=%.6f std=%.6f min=%.6f max=%.6f\n",
           sum/total, sqrtf((float)(sum2/total - (sum/total)*(sum/total))),
           vmin, vmax);

    /* 2-bit Lloyd-Max */
    printf("\n=== 2-bit (4 levels) ===\n");
    float c2[4], b2[3];
    lloyd_max(samples, N_SAMPLES, c2, b2, 4, 200);
    print_array("LM2_CENTROIDS_OPT", c2, 4);
    print_array("LM2_BOUNDARIES_OPT", b2, 3);

    /* 3-bit Lloyd-Max */
    printf("\n=== 3-bit (8 levels) ===\n");
    float c3[8], b3[7];
    lloyd_max(samples, N_SAMPLES, c3, b3, 8, 200);
    print_array("LM3_CENTROIDS_OPT", c3, 8);
    print_array("LM3_BOUNDARIES_OPT", b3, 7);

    /* 4-bit Lloyd-Max */
    printf("\n=== 4-bit (16 levels) ===\n");
    float c4[16], b4[15];
    lloyd_max(samples, N_SAMPLES, c4, b4, 16, 200);
    print_array("LM4_CENTROIDS_OPT", c4, 16);
    print_array("LM4_BOUNDARIES_OPT", b4, 15);

    free(samples);
    return 0;
}
