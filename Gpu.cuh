/*
 * This file is part of the continuous space language and translation model toolkit
 * for statistical machine translation and large vocabulary speech recognition.
 *
 * Copyright 2015, Holger Schwenk, LIUM, University of Le Mans, France
 *
 * The CSLM toolkit is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License version 3 as
 * published by the Free Software Foundation
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
 */


#ifndef _Gpu_cuh
#define _Gpu_cuh

#include "Tools.h"
#include <cublas.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <npps.h>
#include <vector>

#define GPU_CUBLAS_V2
#ifdef GPU_CUBLAS_V2
#include <cublas_v2.h>
#endif

#define CUDA float			// memory on the GPU card 
#define CUDA_SIZE (sizeof(float))	// memory on the GPU card 

extern curandGenerator_t cuda_gen;
extern string cuda_user_list; ///< user specified list of GPUs

/**
 * provides an interface to use Gpu with Cuda
 */
class Gpu
{
public:
  /**
   * initializes Cuda and creates lock files
   * @note selects first device and stream
   * @returns configuration index 0
   */
  static size_t Init();

  /**
   * removes lock-files and deletes all configurations
   */
  static void Unlock();

  /**
   * creates a new Gpu stream on next device
   * @note selects the next device and the new stream
   * @returns new configuration index
   */
  static size_t NewConfig();

  /**
   * gets current configuration index
   */
  static inline size_t GetConfig() {
    return Gpu::curConfIndex; }

  /**
   * sets current device and stream
   * @param stCfg index of configuration to use
   */
  static inline void SetConfig(size_t stCfg) {
    if (stCfg != Gpu::curConfIndex) ChangeConfig(stCfg % Gpu::vConfigs.size()); }

  /**
   * gets number of devices
   */
  static inline size_t GetDeviceCount() {
    return vDevices.size(); }

  /**
   * gets device index
   * @param stCfg index of configuration
   */
  static inline size_t GetDevice(size_t stCfg) {
    return Gpu::vConfigs[stCfg % Gpu::vConfigs.size()].devId; }

  /**
   * sets current device with default stream
   * @param stDevId device index
   */
  static void SetDevice(size_t stDevId);

  /**
   * gets Cuda device number
   * @param stDevId device index
   */
  static inline int GetCudaDevice(size_t stDevId) {
    return Gpu::vDevices[stDevId % Gpu::vDevices.size()].number; }

  /**
   * gets device properties
   * @param stCfg index of configuration
   */
  static inline const cudaDeviceProp& GetDeviceProp(size_t stCfg) {
    return Gpu::vDevices[Gpu::vConfigs[stCfg % Gpu::vConfigs.size()].devId].props; }

  /**
   * allocates memory on Gpu and checks error
   * @param dim data size
   * @param msg message to print in case of error
   * @returns pointer to memory block, or NULL in case of error
   */
  static REAL* Alloc(int dim, const char* msg);

  /**
   * copies data between host and Gpu
   * @param dst destination memory address
   * @param src source memory address
   * @param count size in bytes to copy
   * @param kind type of transfer
   * @returns error code
   */
  static inline cudaError_t MemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    return cudaMemcpyAsync(dst, src, count, kind, Gpu::curStream); }

  /**
   * copies data between host and Gpu
   * @param dst destination memory address
   * @param dpitch pitch of destination memory
   * @param src source memory address
   * @param spitch pitch of source memory
   * @param width width of matrix transfer (columns in bytes)
   * @param height height of matrix transfer (rows)
   * @param kind type of transfer
   * @returns error code
   */
  static inline cudaError_t Memcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) {
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, Gpu::curStream); }

  /**
   * initializes or sets Gpu memory to a value
   * @param devPtr pointer to Gpu memory
   * @param value value to set for each byte of specified memory
   * @param count size in bytes to set
   * @returns error code
   */
  static inline cudaError_t MemsetAsync(void* devPtr, int value, size_t count) {
    return cudaMemsetAsync(devPtr, value, count, Gpu::curStream); }

  /**
   * checks if streams are used concurrently within a device
   * @note depends on the number of new configurations
   */
  static inline bool UseConcurrentStreams() {
    return Gpu::useConcurrentStreams;
  }

  /**
   * waits for current stream tasks to complete
   * @returns error code
   */
  static inline cudaError_t StreamSynchronize() {
    return cudaStreamSynchronize(Gpu::curStream); }

  /**
   * checks error
   * @param msg message to print in case of error
   */
  static void CheckError(const char* msg);


  /** Blas methods */

  static inline void CublasScopy(int n, const REAL* x, int incx, REAL* y, int incy) {
#ifdef GPU_CUBLAS_V2
    cublasScopy(Gpu::curCbHandle, n, x, incx, y, incy); }
#else
    cublasScopy(                  n, x, incx, y, incy); }
#endif

  static inline REAL CublasSasum(int n, const REAL* x, int incx) {
#ifdef GPU_CUBLAS_V2
    REAL result; cublasSasum(Gpu::curCbHandle, n, x, incx, &result); return result; }
#else
    return       cublasSasum(                  n, x, incx); }
#endif

  static inline void CublasSaxpy(int n, REAL alpha, const REAL* x, int incx, REAL* y, int incy) {
#ifdef GPU_CUBLAS_V2
    cublasSaxpy(Gpu::curCbHandle, n, &alpha, x, incx, y, incy); }
#else
    cublasSaxpy(                  n,  alpha, x, incx, y, incy); }
#endif

  static inline void CublasSscal(int n, REAL alpha, REAL* x, int incx) {
#ifdef GPU_CUBLAS_V2
    cublasSscal(Gpu::curCbHandle, n, &alpha, x, incx); }
#else
    cublasSscal(                  n,  alpha, x, incx); }
#endif

  static inline void CublasSgemv(char trans, int m, int n, REAL alpha, const REAL* A, int lda, const REAL* x, int incx, REAL beta, REAL* y, int incy) {
#ifdef GPU_CUBLAS_V2
    cublasOperation_t co = ((trans == 'N') ? CUBLAS_OP_N : ((trans == 'T') ? CUBLAS_OP_T : CUBLAS_OP_C));
    cublasSgemv(Gpu::curCbHandle, co, m, n, &alpha, A, lda, x, incx, &beta, y, incy); }
#else
    cublasSgemv(               trans, m, n,  alpha, A, lda, x, incx,  beta, y, incy); }
#endif

  static inline void CublasSger(int m, int n, REAL alpha, const REAL* x, int incx, const REAL* y, int incy, REAL* A, int lda) {
#ifdef GPU_CUBLAS_V2
    cublasSger(Gpu::curCbHandle, m, n, &alpha, x, incx, y, incy, A, lda); }
#else
    cublasSger(                  m, n,  alpha, x, incx, y, incy, A, lda); }
#endif

  static inline void CublasSgemm(char transa, char transb, int m, int n, int k, REAL alpha, const REAL* A, int lda, const REAL* B, int ldb, REAL beta, REAL* C, int ldc) {
#ifdef GPU_CUBLAS_V2
    cublasOperation_t coa = ((transa == 'N') ? CUBLAS_OP_N : ((transa == 'T') ? CUBLAS_OP_T : CUBLAS_OP_C));
    cublasOperation_t cob = ((transb == 'N') ? CUBLAS_OP_N : ((transb == 'T') ? CUBLAS_OP_T : CUBLAS_OP_C));
    cublasSgemm(Gpu::curCbHandle, coa, cob, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc); }
#else
    cublasSgemm(            transa, transb, m, n, k,  alpha, A, lda, B, ldb,  beta, C, ldc); }
#endif


  /* methods used to launch kernels on Gpu */

  static void MachTabForw(const int bsize, const int odim, REAL *gpu_data_in, REAL *gpu_t, REAL *gpu_data_out);

  static void MachTabBackw(const REAL lrate, const int bsize, const int odim, REAL *gpu_data_in, REAL *gpu_t, REAL *gpu_grad_out);

  static void MachSoftmaxForw(const int bsize, const int odim, REAL *gpu_data_out);
  static void MachSoftmaxStableForw(const int bsize, const int odim, REAL *gpu_data_out);

  static void LinRectifForw(const int n, REAL *gpu_data_out);
  static void LinRectifBackw(const int n, REAL *gpu_data_out, REAL *gpu_grad_out);

  static void DropOut(const int n, REAL *gpu_vect, REAL *rand, REAL thresh);

  static REAL ErrFctSoftmClassError(const int bsize, const int n_classes, REAL *gpu_class_out, REAL *gpu_class_target);

  static REAL ErrFctSoftmCrossEntNgramCalcValue(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_target);
  static REAL ErrFctSoftmCrossEntNgramCalcValueNull(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_target);
  static void ErrFctSoftmCrossEntNgramCalcValueBatch(const int eff_bsize, const int dim, REAL *output, REAL *target, REAL *res);
  // not used anymore
  // static REAL ErrFctSoftmCrossEntNgramCalcValueNth(const int idx, const int odim, REAL *gpu_data_out, REAL *gpu_target);
  static void ErrFctSoftmCrossEntNgramCalcMax(const int eff_bsize, const int dim, REAL *output, REAL *target, REAL *res, int *pos);
  static void ErrFctSoftmCrossEntNgramCalcGrad(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_grad, REAL *gpu_target, REAL * gpu_res);
  static void ErrFctSoftmCrossEntNgramCalcGradNull(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_grad, REAL *gpu_target, REAL * gpu_res);
  static void ErrFctSoftmCrossEntNgramCalcGradCumul(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_grad, REAL *gpu_target);

  static REAL ErrFctSoftmCrossEntNgramMultiCalcGrad(const int bsize, const int dim, const int nb, REAL *gpu_data_out, REAL *gpu_grad, REAL *gpu_target);

  static void MachSoftmaxClassLinForw(const int bsize, const int idim, const int odim, REAL* input, REAL* weights, REAL* bias, REAL* output, int* class_info, const int max_size);

  static void MachSoftmaxClassSoftmForw(const int bsize, const int odim, REAL* gpu_data_out, int* class_info, const int max_size, const int stable);
  static void ErrFctSoftmClassCrossEntNgramCalcGrad(const int bsize, const int odim, REAL* gpu_data_out, REAL* gpu_grad, REAL* gpu_target, int* class_info, REAL* gpu_res);
  static void MachSoftmaxClassLinGradIn(const int bsize, const int idim, const int odim, REAL* grad_out, REAL* weights, REAL* grad_in, int* class_info, const int max_size);
  static void MachSoftmaxClassLinGradUpdate(const int bsize, const int idim, const int odim, REAL* input, REAL* grad_out, REAL* weights, REAL* bias, int* class_info, const int max_size, const REAL lrate, const REAL wdecay);

  static void CopyVectorToMatrix(REAL * mat, REAL * vec, const int M, const int N);
  static void CopyMatrixToMatrixStrided(REAL * dst, REAL * src, const int M, const int N, const int row_stride);
  static void CopyMatrixStridedToMatrix(REAL * dst, REAL * src, const int M, const int N, const int row_stride);

  static void BatchedAXPY(const int n, const REAL a, REAL * x, const int incx, REAL * y, const int incy, const int nb_batch);

  static void ElemwiseExp(const int size, REAL *gpu_data_in, REAL *gpu_data_out);
  static void ElemwiseTanh(const int size, REAL *gpu_data_in, REAL *gpu_data_out);
  static void ElemwiseTanhGrad(const int size, REAL *gpu_data_out, REAL *gpu_grad_out, REAL* gpu_grad_in);

  static void MemSet(REAL *adr, REAL val, int len);

  static void ResSet(REAL val);
  static REAL ResGet();


private:
  /**
   * Gpu device
   */
  struct Device {
    int number;           ///< Gpu device number
    cudaDeviceProp props; ///< device properties
  };

  /**
   * Gpu configuration
   */
  struct Config {
    size_t devId; ///< device index
    cudaStream_t stream; ///< Gpu stream
#ifdef GPU_CUBLAS_V2
    cublasHandle_t cbHandle; ///< Cublas handle
#endif
  };

  static size_t curDevIndex;          ///< current device index
  static size_t curConfIndex;         ///< current configuration index
  static cudaStream_t curStream;      ///< current stream
  static bool useConcurrentStreams;   ///< status of concurrent streams
#ifdef GPU_CUBLAS_V2
  static cublasHandle_t curCbHandle;  ///< current Cublas handle
#endif
  static cudaDeviceProp* curDevProps; ///< current device properties
  static std::vector<Gpu::Device> vDevices; ///< vector of Gpu devices to be used
  static std::vector<Gpu::Config> vConfigs; ///< vector of Gpu configurations

  /**
   * changes current configuration
   * @param stCfg index of configuration to use
   */
  static void ChangeConfig(size_t stCfg);

};

#endif
