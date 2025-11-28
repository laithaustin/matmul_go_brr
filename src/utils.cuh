#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ==================== CUDA Operations ====================

void cudaCheck(cudaError_t error, const char *file, int line);

#define CUDA_CHECK(err) (cudaCheck(err, __FILE__, __LINE__))

void CudaDeviceInfo();

// ==================== Matrix Operations ====================

void randomize_matrix(float *mat, int N);

void copy_matrix(float *src, float *dest, int N);

void print_matrix(const float *A, int M, int N, std::ofstream &fs);

bool verify_matrix(float *matRef, float *matOut, int N);

// ==================== Timing Operations ====================

float get_sec();

float cpu_elapsed_time(float &beg, float &end);

// ==================== Kernel Operations ====================

void test_kernel(int kernel_num, int M, int N, int K, float alpha, 
                float *A, float *B, float beta, float *C,
                cublasHandle_t handle);
