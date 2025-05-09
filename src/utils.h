#ifndef UTILS_H
#define UTILS_H
// Base template for 1D arrays
template <size_t N>
void printr(double (*arr)[N]) {
    for (size_t i = 0; i < N; i++) {
        printf("%f, ", arr[i]);
    }
    printf("\n");
}

// Specialization for 2D arrays
template <size_t N, size_t M>
void printr(double (*arr)[N][M]) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            printf("%f, ", arr[i][j]);
        }
        printf("\n");
    }
}

// Specialization for 3D arrays
template <size_t N, size_t M, size_t O>
void printr(double (*arr)[N][M][O]) {
    for (size_t i = 0; i < N; i++) {
        printf("Slice %zu:\n", i);
        for (size_t j = 0; j < M; j++) {
            for (size_t k = 0; k < O; k++) {
                printf("%f, ", arr[i][j][k]);
            }
            printf("\n");
        }
    }
}

// Specialization for 4D arrays
template <size_t N, size_t M, size_t O, size_t P>
void printr(double (*arr)[N][M][O][P]) {
    for (size_t i = 0; i < N; i++) {
        printf("Block %zu:\n", i);
        for (size_t j = 0; j < M; j++) {
            printf("  Slice %zu:\n", j);
            for (size_t k = 0; k < O; k++) {
                for (size_t l = 0; l < P; l++) {
                    printf("%f, ", arr[i][j][k][l]);
                }
                printf("\n");
            }
        }
    }
}

#endif