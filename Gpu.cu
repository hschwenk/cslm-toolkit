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

using namespace std;
#include <algorithm>
#include <map>
#include <sstream>
#include <signal.h>
#define RAISE raise(SIGINT);

typedef float REAL;
#define NULL_WORD (-1)		// from WordList.h
#define LOG_PROBA_NONE 999	// from ErrFact.h
#define LOCK_FNAME "/tmp/gpu_lock.pid%d.gpu%d"
#define LOCK_FNAME_LEN 256	// Hack ;-)

#include <npps.h>
#include <cublas.h>
#include <cuda_runtime_api.h>
#include <nppcore.h>
#include "nvml.h"
#include "Gpu.cuh"
#include "Tools.h" //For Error()


// global variables
curandGenerator_t cuda_gen;
string cuda_user_list;	// user specified list of GPUs
static REAL *gpu_result;  
#define GPU_BUF_DIM 65536
static REAL *gpu_buf;  

size_t Gpu::curDevIndex  = (size_t)-1;   ///< current device index
size_t Gpu::curConfIndex = (size_t)-1;   ///< current configuration index
cudaStream_t Gpu::curStream = NULL;      ///< current stream
bool Gpu::useConcurrentStreams = false;  ///< status of concurrent streams
#ifdef GPU_CUBLAS_V2
cublasHandle_t Gpu::curCbHandle = NULL;  ///< current Cublas handle
#endif
cudaDeviceProp* Gpu::curDevProps = NULL; ///< device properties
vector<Gpu::Device> Gpu::vDevices; ///< vector of Gpu devices to be used
vector<Gpu::Config> Gpu::vConfigs; ///< vector of Gpu configurations

void HandlerSigTERM(int s)
{
  printf("Catched signal: removing lock-files\n");
  Gpu::Unlock();
  exit(1);
}

/**
 * initializes Cuda and creates lock files
 * @note selects first device and stream
 * @returns configuration index 0
 */
size_t Gpu::Init()
{
  size_t stId = 0;
  if (0 >= Gpu::vConfigs.size()) {
    Gpu::vConfigs.resize(1);

    cout << "Initializing Nvidia GPU card" << endl;
    int dev_max = 0;
    cudaGetDeviceCount(&dev_max);
    bool bSelAuto = (':' != cuda_user_list[0]);
    Gpu::Device dev;
    if (0 < dev_max) {
      if (1 == dev_max)
        cout << " - found 1 card:" << endl;
      else
        cout << " - found " << dev_max << " cards:" << endl;
      if (bSelAuto)
        nvmlInit();
      nvmlDevice_t nd;
      nvmlUtilization_t nu;
      multimap<uint,Gpu::Device> mSelDev;
      for (dev.number = 0 ; dev.number < dev_max ; dev.number++) {
        cudaGetDeviceProperties(&dev.props, dev.number);
        int nb_cores_per_multiprocessor = -1;
        if(dev.props.major == 1 && (dev.props.minor == 0||dev.props.minor == 1||dev.props.minor == 2||dev.props.minor == 3))
            nb_cores_per_multiprocessor = 8;
        else if(dev.props.major == 2 && dev.props.minor == 0)
            nb_cores_per_multiprocessor = 32;
        else if(dev.props.major == 2 && dev.props.minor == 1)
            nb_cores_per_multiprocessor = 48;
        else if(dev.props.major == 3 && (dev.props.minor == 0||dev.props.minor == 5))
            nb_cores_per_multiprocessor = 192;


        printf("    %d: %s with %d CPUs x %d threads running at %4.2f Ghz, %d MBytes of memory, use -arch=sm_%d%d",
          dev.number, dev.props.name, dev.props.multiProcessorCount, nb_cores_per_multiprocessor,
          dev.props.clockRate/1000000.0, (int) (dev.props.totalGlobalMem/1024/1024),
          dev.props.major, dev.props.minor);
        if (bSelAuto) {
          if (   (nvmlDeviceGetHandleByIndex(dev.number, &nd) == NVML_SUCCESS)
              && (nvmlDeviceGetUtilizationRates( nd    , &nu) == NVML_SUCCESS) )
            printf(", utilization %d%%", nu.gpu);
          mSelDev.insert(make_pair(nu.gpu, dev));
        }
        printf("\n");
      }
      if (bSelAuto) { // select devices automatically
        nvmlShutdown();
        int iMaxDev = std::min(std::max(atoi(cuda_user_list.c_str()), 0), dev_max);
        for (multimap<uint,Gpu::Device>::const_iterator mmci = mSelDev.begin() ; 0 < iMaxDev-- ; mmci++)
          Gpu::vDevices.push_back(mmci->second);
      }
    }

    if (!bSelAuto) { // read devices specified by user
      char c;
      istringstream iss;
      iss.str(cuda_user_list);
      while (iss.good()) {
        iss >> c >> dev.number;
        Gpu::vDevices.push_back(dev);
        cudaGetDeviceProperties(&Gpu::vDevices.back().props, dev.number);
      }
      if (iss.fail())
        ErrorN("format error in the selection of CUDA devices \"%s\"", cuda_user_list.c_str() + 1);
    }
    size_t dev_sel = Gpu::vDevices.size();
    switch (dev_sel) {
      case 0: printf(" - no GPU device selected\n");
              dev.number = 0;
              Gpu::vDevices.push_back(dev);
              dev_sel = 1;
              cudaGetDeviceProperties(&Gpu::vDevices.back().props, dev.number);
      case 1: printf(" - using device %d\n", Gpu::vDevices[0].number);
              cudaSetDevice(Gpu::vDevices[0].number);
              break;
      default:
        if (dev_sel > (size_t)dev_max) {
          printf(" - requested more GPU devices than available, using %d first ones\n", dev_max);
          dev_sel = dev_max;
          Gpu::vDevices.resize(dev_sel);
        }
        printf(" - using %lu devices in parallel:", dev_sel);
        for (size_t d = 0 ; d < dev_sel ; d++) {
          int n = Gpu::vDevices[d].number;
          printf(" %d", n);
          if ((n < 0) || (n >= dev_max))
            Error("illegal device identifier");
        }
        printf("\n");
        cudaSetDevice(Gpu::vDevices[0].number);
    }

    // initialize cublas and random generator
    cublasInit();
    Gpu::CheckError("initialization of card\n");
    curandCreateGenerator(&cuda_gen, CURAND_RNG_PSEUDO_DEFAULT);
    // curandSetPseudoRandomGeneratorSeed(cuda_gen, CUDA_SEED);
    Gpu::CheckError("initialization of random generator\n");

    // allocate buffers
    gpu_buf = Gpu::Alloc(GPU_BUF_DIM*sizeof(REAL),"internal buffer on GPU");

    // locking devices
    ofstream lfs;
    char lfname[LOCK_FNAME_LEN] = LOCK_FNAME;
    for (size_t d = 0 ; d < dev_sel ; d++) {
      sprintf(lfname, LOCK_FNAME, getpid(), Gpu::vDevices[d].number);
      lfs.open(lfname,ios::out);
      CHECK_FILE(lfs, lfname);
      lfs << "Runing job " << getpid() << " on GPU " << Gpu::vDevices[d].number << endl;
      lfs.close();
    }

    // catch signals to clean up lock-files
    signal(SIGINT , HandlerSigTERM);
    signal(SIGHUP , HandlerSigTERM);
    signal(SIGFPE , HandlerSigTERM);
    signal(SIGSEGV, HandlerSigTERM);
    signal(SIGTERM, HandlerSigTERM);

    // create default configuration
    Gpu::Config& newConfig = Gpu::vConfigs.back();
    Gpu::curDevIndex = newConfig.devId = 0;
    Gpu::curConfIndex = stId;
    newConfig.stream = NULL;
#ifdef GPU_CUBLAS_V2
    cublasCreate(&newConfig.cbHandle);
    Gpu::curCbHandle = newConfig.cbHandle;
#endif
    Gpu::curDevProps = &Gpu::vDevices[0].props;
  }
  return stId;
}

/**
 * removes lock-files and deletes all configurations
 */
void Gpu::Unlock()
{
  // remove lock-files
  Gpu::curDevIndex = (size_t)-1;
  char lfname[LOCK_FNAME_LEN] = LOCK_FNAME;
  for (std::vector<Gpu::Device>::iterator id = Gpu::vDevices.begin() ; id != Gpu::vDevices.end() ; id++) {
    sprintf(lfname, LOCK_FNAME, getpid(), id->number);
    if (unlink(lfname))
      cerr << " - ERROR: removing lock file " << lfname << endl;
  }

  // destroy streams
  Gpu::curConfIndex = (size_t)-1;
  Gpu::curStream = NULL;
  Gpu::useConcurrentStreams = false;
#ifdef GPU_CUBLAS_V2
  Gpu::curCbHandle = NULL;
#endif
  Gpu::curDevProps = NULL;
  Gpu::vDevices.clear();
  for (std::vector<Gpu::Config>::iterator igc = Gpu::vConfigs.begin() ; igc != Gpu::vConfigs.end() ; igc++) {
    if (NULL != igc->stream)
      cudaStreamDestroy(igc->stream);
#ifdef GPU_CUBLAS_V2
    if (NULL != igc->cbHandle)
      cublasDestroy(igc->cbHandle);
#endif
  }
  Gpu::vConfigs.clear();
}


/**
 * creates a new Gpu stream on next device
 * @note selects the next device and the new stream
 * @returns new configuration index
 */
size_t Gpu::NewConfig()
{
  size_t stId = Gpu::vConfigs.size();
  if (0 < stId) {
    Gpu::useConcurrentStreams |= (Gpu::vDevices.size() <= (0.8 * (stId + 1)));
    Gpu::vConfigs.resize(stId + 1);
    Gpu::Config& newConfig = Gpu::vConfigs.back();
    newConfig.devId = ((Gpu::curDevIndex + 1) % Gpu::vDevices.size());
    newConfig.stream = NULL;
#ifdef GPU_CUBLAS_V2
    newConfig.cbHandle = NULL;
#endif
    Gpu::ChangeConfig(stId);
    return stId;
  }
  else
    return Gpu::Init();
}

/**
 * changes current configuration
 * @param stCfg index of configuration to use
 */
void Gpu::ChangeConfig(size_t stCfg)
{
  Gpu::curConfIndex = stCfg;
  Gpu::Config& config = Gpu::vConfigs[Gpu::curConfIndex];
  if (Gpu::curDevIndex != config.devId) {
    Gpu::curDevIndex = config.devId;
    cudaSetDevice(Gpu::vDevices[Gpu::curDevIndex].number);
    Gpu::curDevProps = &Gpu::vDevices[Gpu::curDevIndex].props;
  }
#ifdef GPU_CUBLAS_V2
  if (NULL == config.cbHandle)
    cublasCreate(&config.cbHandle);
  if (Gpu::useConcurrentStreams && (NULL == config.stream)) {
    cudaStreamSynchronize(NULL);
    cudaStreamCreate(&config.stream);
    cublasSetStream(config.cbHandle, config.stream);
  }
  if (Gpu::curStream != config.stream) {
    Gpu::curStream = config.stream;
    nppSetStream(Gpu::curStream);
  }
  Gpu::curCbHandle = config.cbHandle;
  debug4("Gpu::ChangeConfig cfg=%zu dev=%d str=%x cbh=%x\n", Gpu::curConfIndex, Gpu::vDevices[Gpu::curDevIndex].number, Gpu::curStream, Gpu::curCbHandle);
#endif
}

/**
 * sets current device with default stream
 * @param stDevId device index
 */
void Gpu::SetDevice(size_t stDevId)
{
  Gpu::curConfIndex = (size_t)-1;
  if (Gpu::curDevIndex != stDevId) {
    Gpu::curDevIndex = (stDevId % Gpu::vDevices.size());
    cudaSetDevice(Gpu::vDevices[Gpu::curDevIndex].number);
    Gpu::curDevProps = &Gpu::vDevices[Gpu::curDevIndex].props;
  }
#ifdef GPU_CUBLAS_V2
  if (NULL != Gpu::curStream) {
    Gpu::curStream = NULL;
    nppSetStream(Gpu::curStream);
  }
  Gpu::curCbHandle = NULL;
#endif
}

/**
 * allocates memory on Gpu and checks error
 * @param msg message to print in case of error
 */
REAL* Gpu::Alloc(int dim, const char* msg) {
  void* gpu_mem;
  char err_msg[1024];
  sprintf(err_msg, "CUDA: can't allocate memory for %s", msg);
  sprintf(err_msg, "CUDA: can't allocate memory (%dMB) for %s", (int)(dim / 1024 / 1024 * sizeof(REAL)), msg);
  if (dim > 0) {
    cublasAlloc(dim, CUDA_SIZE, &gpu_mem);
#ifdef DEBUG
    int dev = -1;
    cudaGetDevice(&dev);
    debug3("allocated %ld at %p on device %d\n",  dim * CUDA_SIZE, gpu_mem, dev);
#endif
    Gpu::CheckError(err_msg);
    if (NULL == gpu_mem)
      Error(err_msg);
    return (CUDA*)gpu_mem;
  }
  else
    return NULL;
}

/**
 * checks error
 * @param msg message to print in case of error
 */
void Gpu::CheckError(const char* msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    ErrorN("CUDA: ERROR %d in %s: %s\n", cublasGetError(), msg, cudaGetErrorString(err));
}


// Corresponds to 2.0*numeric_limits<float>::min()
__device__ REAL GPU_LOG_LOWER_BOUND = 2.35099e-38;
__device__ REAL gpu_safelog(REAL x) { return (x<GPU_LOG_LOWER_BOUND) ? log(GPU_LOG_LOWER_BOUND) : log(x); };


//-----------------------------------------------
// forward pass for MachTab
//-----------------------------------------------

__global__
void KernelMachTabForw(const int bsize, const int odim, REAL *gpu_data_in, REAL *gpu_t, REAL *gpu_data_out)
{
  for (int b=blockIdx.x ; b<bsize ; b+=gridDim.x) {
    int idx= (int) gpu_data_in[b];
    int offso=b*odim;
    int offst=idx*odim;
    for (int i=threadIdx.x ; i<odim ; i+=blockDim.x) {
      if (idx==NULL_WORD) gpu_data_out[i+offso] = 0.0;
                     else gpu_data_out[i+offso] = gpu_t[i+offst];
    }
  }
}

void Gpu::MachTabForw(const int bsize, const int odim,
		    REAL *gpu_data_in, REAL *gpu_t, REAL *gpu_data_out)
{
  int n_threads = std::min(Gpu::curDevProps->maxThreadsDim[0], odim);
  int n_blocks = std::min(Gpu::curDevProps->maxGridSize[0], bsize);
  KernelMachTabForw<<<n_blocks, n_threads, 0, Gpu::curStream>>>(bsize, odim, gpu_data_in, gpu_t, gpu_data_out);
}


//-----------------------------------------------
// backward pass for MachTab
//-----------------------------------------------

__global__
void KernelMachTabBackw(const REAL lrate, const int bsize, const int odim,
                        REAL *gpu_data_in, REAL *gpu_t, REAL *gpu_grad_out)
{
  for (int b=blockIdx.x; b<bsize; b+=gridDim.x) {
    for (int i=threadIdx.x; i<odim; i+=blockDim.x) {
      int idx = (int) gpu_data_in[b];
      // Use atomicAdd instead of += to avoid race conditions between threads
      if (idx != NULL_WORD)
        atomicAdd(gpu_t+i+idx*odim, lrate * gpu_grad_out[i+b*odim]);
    }
  }
}

void Gpu::MachTabBackw(const REAL lrate, const int bsize, const int odim,
                     REAL *gpu_data_in, REAL *gpu_t, REAL *gpu_grad_out)
{
  int n_threads = std::min(Gpu::curDevProps->maxThreadsDim[0], odim);
  int n_blocks = std::min(Gpu::curDevProps->maxGridSize[0], bsize);
  KernelMachTabBackw<<<n_blocks, n_threads, 0, Gpu::curStream>>>(lrate, bsize, odim, gpu_data_in, gpu_t, gpu_grad_out);
}


//-----------------------------------------------
// Softmax normalization
//-----------------------------------------------

__global__ void KernelSoftmax(int M, int N,
			      const REAL * x, const int sx0, const int sx1,
 			      REAL * sm, const int sm_s0, const int sm_s1)
{
  extern __shared__ REAL buf[];
  for (int blockIDX = blockIdx.x; blockIDX < M; blockIDX += gridDim.x) {
    REAL sum = 0;
#pragma unroll 16
    for (int i = threadIdx.x; i< N; i += blockDim.x){
      sum += exp(x[blockIDX * sx0 + i * sx1]);
    }
    buf[threadIdx.x] = sum;
    __syncthreads();

    // This function trashes buf[1..warpsize], leaving the reduction result in buf[0].
    if (threadIdx.x < warpSize){
#pragma unroll 8
      for (int i = threadIdx.x + warpSize; i < blockDim.x; i += warpSize){
                buf[threadIdx.x] += buf[i];
      }
      if (threadIdx.x < 16){
                //reduce so that threadIdx.x 0 has the sum of everything
                if(threadIdx.x + 16 < N)
                    buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+16];
                if(threadIdx.x + 8 < N)
                    buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+8];
                if(threadIdx.x + 4 < N)
                    buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+4];
                if(threadIdx.x + 2 < N)
                    buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+2];
                if(threadIdx.x + 1 < N)
                    buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+1];
      }
    }
    __syncthreads();
    REAL row_sum = buf[0];
#pragma unroll 16
    for (int i = threadIdx.x; i< N; i += blockDim.x){
      sm[blockIDX * sm_s0 + i * sm_s1] = exp(x[blockIDX * sx0 + i * sx1]) / row_sum;
    }
    __syncthreads();
  }
}
void Gpu::MachSoftmaxForw(const int bsize, const int odim, REAL *gpu_data_out)
{
  if(0){
    //This is the original code that is know to work correctly in all case,
    //But is slower.
    nppsExp_32f_I(gpu_data_out, bsize*odim);

    REAL sum, *optr=gpu_data_out;

    for (int b=0; b<bsize; b++,optr+=odim) {
      sum=Gpu::CublasSasum(odim,optr,1);  // exp(x) is always positive -> we can use the sum_i (ABS(x_i))
      nppsMulC_32f_I(1.0/sum,optr,odim);
    }
    return;
  }

  //int warpSize = 32;
//The follwing check need to access the GPU properties to do it.
//To don't do this access each time, we have done it in MachSoftmax.cpp
//  if(warpSize != 32){
//    Error("Gpu::MachSoftmaxForw suppose the warpSize is 32. If run with a GPU with other warpSize"
//	  " like the current GPU, it will return wrong Results. You must update the reduction in KernelSoftmax");
//  }
  int n_blocks = std::min(bsize, 32 * 1024);
  int n_threads = std::min(odim, 512);
  int n_shared_bytes = n_threads * sizeof(REAL);
  if (bsize > 0){
    KernelSoftmax<<<n_blocks, n_threads, n_shared_bytes, Gpu::curStream>>>(
                            bsize,
                            odim,
                            gpu_data_out,
                            odim, //x.stride[0
                            1, //x.stride[1]
                            gpu_data_out,
                            odim, //sm.stride[0]
                            1//sm.stride[1]
                    );
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err){
      printf("KernelSoftmax: n_blockn=%d, n_threads=%d, n_shared_bytes=%d odim=%d\n",
             n_blocks, n_threads, n_shared_bytes, odim);
      Error(cudaGetErrorString(err));
    }
  }
}

//-----------------------------------------------
// Softmax stable normalization
//-----------------------------------------------

__global__ void KernelSoftmaxStable(int M, int N,
                                     const REAL * x, const int sx0, const int sx1,
                                     REAL * sm, const int sm_s0, const int sm_s1)
{
  extern __shared__ REAL buf[];
  for (int blockIDX = blockIdx.x; blockIDX < M; blockIDX += gridDim.x) {
    REAL max_ = x[blockIDX * sx0 + threadIdx.x * sx1];
    for (int i = threadIdx.x + blockDim.x; i< N; i += blockDim.x) {
      max_ = max(max_, x[blockIDX * sx0 + i * sx1]);
    };
    buf[threadIdx.x] = max_;
    __syncthreads();

    // This function trashes buf[1..n_threads], leaving the reduction result in buf[0].
    // Find the max to stabilize the softmax
    if (threadIdx.x < warpSize)
    {
      for (int i = threadIdx.x + warpSize; i < blockDim.x; i += warpSize) {
                buf[threadIdx.x] = max(buf[threadIdx.x], buf[i]);
      }
      if (threadIdx.x < 16) {
                //reduce so that threadIdx.x 0 has the max of everything
                if(threadIdx.x + 16 < N)
                    buf[threadIdx.x] = max(buf[threadIdx.x], buf[threadIdx.x+16]);
                if(threadIdx.x + 8 < N)
                    buf[threadIdx.x] = max(buf[threadIdx.x], buf[threadIdx.x+8]);
                if(threadIdx.x + 4 < N)
                    buf[threadIdx.x] = max(buf[threadIdx.x], buf[threadIdx.x+4]);
                if(threadIdx.x + 2 < N)
                    buf[threadIdx.x] = max(buf[threadIdx.x], buf[threadIdx.x+2]);
                if(threadIdx.x + 1 < N)
                    buf[threadIdx.x] = max(buf[threadIdx.x], buf[threadIdx.x+1]);
            }
    }

    __syncthreads();
    REAL row_max = buf[0];
    __syncthreads();
    REAL sum = 0;
    for(int i=threadIdx.x; i<N; i+=blockDim.x){
      sum += exp(x[blockIDX * sx0 + i * sx1] - row_max);
    };
    buf[threadIdx.x] = sum; 
    __syncthreads();

    // This function trashes buf[1..N], leaving the reduction result in buf[0].
    if (threadIdx.x < warpSize){
      for (int i = threadIdx.x + warpSize; i < blockDim.x; i += warpSize){
                buf[threadIdx.x] += buf[i];
      }
      if (threadIdx.x < 16){
                //reduce so that threadIdx.x 0 has the sum of everything
                if(threadIdx.x + 16 < N)
                    buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+16];
                if(threadIdx.x + 8 < N)
                    buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+8];
                if(threadIdx.x + 4 < N)
                    buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+4];
                if(threadIdx.x + 2 < N)
                    buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+2];
                if(threadIdx.x + 1 < N)
                    buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+1];
      }
    }
    __syncthreads();
    REAL row_sum = buf[0];
    for (int i = threadIdx.x; i< N; i += blockDim.x){
      sm[blockIDX * sm_s0 + i * sm_s1] = exp(x[blockIDX * sx0 + i * sx1] - row_max) / row_sum;
    }
    __syncthreads();
  }
}

void Gpu::MachSoftmaxStableForw(const int bsize, const int odim, REAL *gpu_data_out)
{
  if(0){
    Error("Not implemented!");
    //This is the original code that is know to work correctly in all case,
    //But is slower.
    nppsExp_32f_I(gpu_data_out, bsize*odim);

    REAL sum, *optr=gpu_data_out;

    for (int b=0; b<bsize; b++,optr+=odim) {
      sum=Gpu::CublasSasum(odim,optr,1);  // exp(x) is always positive -> we can use the sum_i (ABS(x_i))
      nppsMulC_32f_I(1.0/sum,optr,odim);
    }
    return;
  }
  //int warpSize = 32;
//The follwing check need to access the GPU properties to do it.
//To don't do this access each time, we have done it in MachSoftmaxStable.cpp
//  if(warpSize != 32){
//    Error("Gpu::MachSoftmaxStableForw suppose the warpSize is 32. If run with a GPU with other warpSize"
//        " like the current GPU, it will return wrong Results. You must update the reduction in KernelSoftmaxStable");
//  }
  int n_blocks = std::min(bsize, 32 * 1024);
  int n_threads = std::min(odim, 512);
  int n_shared_bytes = n_threads * sizeof(REAL);
  if (bsize > 0){
    KernelSoftmaxStable<<<n_blocks, n_threads, n_shared_bytes, Gpu::curStream>>>(
                            bsize,
                            odim,
                            gpu_data_out,
                            odim, //x.stride[0]
                            1, //x.stride[1]
                            gpu_data_out,
                            odim, //sm.stride[0]
                            1//sm.stride[1]
                    );
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err){
      printf("n_blocks=%d, n_threads=%d, n_shared_bytes=%d odim=%d\n",
             n_blocks, n_threads, n_shared_bytes, odim);
      Error(cudaGetErrorString(err));
    }
  }
}

//-----------------------------------------------
// Linear Rectifier units
//-----------------------------------------------

__global__
void KernelLinRectifForw(const int n, REAL *gpu_data_out)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int n_threads = blockDim.x * gridDim.x;
  int id = tx * blockDim.x + bx * gridDim.x;
  for(int i = id; i < n; i += n_threads){
    if (gpu_data_out[i]<0) gpu_data_out[i]=0;
  }
}

void Gpu::LinRectifForw(const int n, REAL *gpu_data_out)
{
  int nb_thread = std::min(n, 256);
  int nb_block = n / 256;
  KernelLinRectifForw<<<nb_block, nb_thread, 0, Gpu::curStream>>>(n, gpu_data_out);
}

__global__
void KernelLinRectifBackw(const int n, REAL *gpu_data_out, REAL *gpu_grad_out)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int n_threads = blockDim.x * gridDim.x;
  int id = tx * blockDim.x + bx * gridDim.x;
  for(int i = id; i < n; i += n_threads){
    if (gpu_data_out[i]<0) gpu_grad_out[i]=0; else gpu_grad_out[i]=1;
  }
}

void Gpu::LinRectifBackw(const int n, REAL *gpu_data_out, REAL *gpu_grad_out)
{
  int nb_thread = std::min(n, 256);
  int nb_block = n / 256;
  KernelLinRectifBackw<<<nb_block, nb_thread, 0, Gpu::curStream>>>(n, gpu_data_out, gpu_grad_out);
}

//-----------------------------------------------
// Helper functions for drop-out
//-----------------------------------------------

__global__
void KernelDropOut(const int n, REAL *gpu_vect, REAL *rand, REAL thresh)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int n_threads = blockDim.x * gridDim.x;
  int id = tx * blockDim.x + bx * gridDim.x;
  for (int i = id; i < n; i += n_threads) {
    if (rand[i]<thresh) gpu_vect[i]=0.0;
  }
}

void Gpu::DropOut(const int n, REAL *gpu_vect, REAL *rand, REAL thresh)
{
  int nb_thread = std::min(n, 256);
  int nb_block = n / 256;
  KernelDropOut<<<nb_block, nb_thread, 0, Gpu::curStream>>>(n, gpu_vect, rand, thresh);
}

//-----------------------------------------------
// ErrFctSoftmCrossEntNgram::CalcValue
//-----------------------------------------------

__global__
void KernelErrFctSoftmCrossEntNgramCalcValue(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_target,
					     REAL *gpu_res)
{
  extern __shared__ REAL buf[];
  REAL err=0.0;
  for (int b=threadIdx.x ; b<bsize ; b+=blockDim.x)
     err += gpu_safelog(gpu_data_out[b*odim + (uint) gpu_target[b]]);
  buf[threadIdx.x] = err;
  __syncthreads();
  if(threadIdx.x == 0) {
    for(int i=1 ; i<blockDim.x ; i++)
      err += buf[i];
    atomicAdd(gpu_res, err);
  }
}


REAL Gpu::ErrFctSoftmCrossEntNgramCalcValue(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_target)
{
  REAL res;
  if (gpu_result==NULL) cudaMalloc(&gpu_result,sizeof(REAL));
  cudaMemsetAsync(gpu_result, 0.0, sizeof(REAL), Gpu::curStream); //Each thread will atomicAdd into it.
  int n_threads = std::min(Gpu::curDevProps->maxThreadsDim[0], bsize);
  KernelErrFctSoftmCrossEntNgramCalcValue<<<1, n_threads, n_threads*sizeof(REAL), Gpu::curStream>>>(bsize, odim, gpu_data_out, gpu_target, gpu_result);
  cudaMemcpyAsync(&res, gpu_result, sizeof(REAL), cudaMemcpyDeviceToHost, Gpu::curStream);
  cudaStreamSynchronize(Gpu::curStream);
  return res;
}

//-----------------------------------------------
// ErrFctSoftmCrossEntNgram::CalcValueNull
//-----------------------------------------------

__global__
void KernelErrFctSoftmCrossEntNgramCalcValueNull(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_target,
					     REAL *gpu_res)
{
  extern __shared__ REAL buf[];
  REAL err=0.0;
  for (int b=threadIdx.x ; b<bsize ; b+=blockDim.x) {
     int tidx = gpu_target[b]; // do not cast to uint ! Otherwise, nvcc will transform the -1 to 0!
     if (tidx != NULL_WORD) err += gpu_safelog(gpu_data_out[b*odim + tidx]);
  }
  buf[threadIdx.x] = err;
  __syncthreads();
  if(threadIdx.x == 0) {
    for(int i=1 ; i<blockDim.x ; i++)
      err += buf[i];
    atomicAdd(gpu_res, err);
  }
}


REAL Gpu::ErrFctSoftmCrossEntNgramCalcValueNull(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_target)
{
  REAL res;
  if (gpu_result==NULL) cudaMalloc(&gpu_result,sizeof(REAL));
  cudaMemsetAsync(gpu_result, 0.0, sizeof(REAL), Gpu::curStream); //Each thread will atomicAdd into it.
  int n_threads = std::min(Gpu::curDevProps->maxThreadsDim[0], bsize);
  KernelErrFctSoftmCrossEntNgramCalcValueNull<<<1, n_threads, n_threads*sizeof(REAL), Gpu::curStream>>>(bsize, odim, gpu_data_out, gpu_target, gpu_result);
  cudaMemcpyAsync(&res, gpu_result, sizeof(REAL), cudaMemcpyDeviceToHost, Gpu::curStream);
  cudaStreamSynchronize(Gpu::curStream);
  return res;
}

//-----------------------------------------------
// ErrFctSoftmCrossEntNgram::CalcValueBatch
//-----------------------------------------------

__global__
void KernelErrFctSoftmCrossEntNgramCalcValueBatch(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_target, REAL *tmp_buf)
{
  //extern __shared__ REAL buf[];
  for (int b=threadIdx.x ; b<bsize ; b+=blockDim.x) {
     int tidx = gpu_target[b]; // do not cast to uint ! Otherwise, nvcc will transform the -1 to 0!
     if (tidx== NULL_WORD)
       tmp_buf[b] = LOG_PROBA_NONE;	// handle NULL_WORD
     else
       tmp_buf[b] = gpu_safelog(gpu_data_out[b*odim + tidx]);
  }
}


void Gpu::ErrFctSoftmCrossEntNgramCalcValueBatch(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_target, REAL *res_vect)
{
  if (odim > GPU_BUF_DIM)
    Error("Gpu::ErrFctSoftmCrossEntNgramCalcValueBatch(): odim (%d) is larger than internal buffer (%d)"); //,odim,GPU_BUF_DIM);
  int n_threads = std::min(Gpu::curDevProps->maxThreadsDim[0], bsize);
  KernelErrFctSoftmCrossEntNgramCalcValueBatch<<<1, n_threads, 0, Gpu::curStream>>>(bsize, odim, gpu_data_out, gpu_target, gpu_buf);
  cudaMemcpyAsync(res_vect, gpu_buf, bsize*sizeof(REAL), cudaMemcpyDeviceToHost, Gpu::curStream);
  cudaStreamSynchronize(Gpu::curStream);
}

//-----------------------------------------------
// ErrFctSoftmCrossEntNgram::CalcMax
//-----------------------------------------------

void Gpu::ErrFctSoftmCrossEntNgramCalcMax(const int eff_bsize, const int dim, REAL *output, REAL *target, REAL *res, int *pos)
{
  Error("TODO: Gpu::ErrFctSoftmCrossEntNgramCalcMax()");
}

#if 0 // not used anymore, use CalcvalueBatch() instead
__global__
void KernelErrFctSoftmCrossEntNgramCalcValueNth(const int idx, const int odim, REAL *gpu_data_out, REAL *gpu_target, REAL *gpu_res)
{
  int tidx = (int) gpu_target[idx]; // do not cast to uint ! Otherwise, nvcc will transform the -1 to 0!
  if (tdx<0) // NULL_WORD 
    *gpu_res=-1;
  else
    *gpu_res = gpu_safelog(gpu_data_out[idx*odim + tidx]);
}


REAL Gpu::ErrFctSoftmCrossEntNgramCalcValueNth(const int idx, const int odim, REAL *gpu_data_out, REAL *gpu_target)
{
  REAL res;
  if (gpu_result==NULL) cudaMalloc(&gpu_result,sizeof(REAL));
  KernelErrFctSoftmCrossEntNgramCalcValueNth<<<1, 1, 1*sizeof(REAL), Gpu::curStream>>>(idx, odim, gpu_data_out, gpu_target, gpu_result);
  cudaMemcpyAsync(&res, gpu_result, sizeof(REAL), cudaMemcpyDeviceToHost, Gpu::curStream);
  cudaStreamSynchronize(Gpu::curStream);
  return res;
#endif


//-----------------------------------------------
// ErrFctSoftmClassCrossEntNgram::CalcWordClassError
//-----------------------------------------------

__global__
void KernelErrFctSoftmClassError(const int bsize, const int n_classes, REAL *gpu_class_out, REAL *gpu_class_target,
                                 REAL *gpu_res)
{
  int class_err=0;
  REAL *ocptr=gpu_class_out;
  REAL *tcptr=gpu_class_target;
  for (int b=0; b<bsize; b++) {
    REAL max_oclass = ocptr[0];
    int argmax = 0;
    for (int i=1; i<n_classes; i++) {
      REAL oclass_i = ocptr[i];
      if (oclass_i > max_oclass) {
        argmax = i;
        max_oclass = oclass_i;
      }
    }
    if ((int) *tcptr != argmax)
      class_err++;

    ocptr += n_classes;
    tcptr++;
  }
  *gpu_res = (REAL) class_err;
}

__global__ void KernelErrFctSoftmClassError2(const int bsize, const int n_classes,
    REAL *gpu_class_out, REAL *gpu_class_target, REAL *gpu_res)
{
  extern __shared__ REAL buf[];
  buf[threadIdx.x] = 0;
  for (int i = threadIdx.x; i < bsize; i += blockDim.x) {
    int argmax = 0;
    REAL max_oclass = gpu_class_out[i*n_classes];
    for (int j = 1; j < n_classes; j++) {
      REAL oclass_j = gpu_class_out[i*n_classes + j];
      if (oclass_j > max_oclass) {
        argmax = j;
        max_oclass = oclass_j;
      }
    }
    if ((int) gpu_class_target[i] != argmax)
      buf[threadIdx.x] += 1;
  }
  __syncthreads();
  // Reduce sum into buf[0]
  if (threadIdx.x < warpSize) {
    for (int i = threadIdx.x + warpSize; i < blockDim.x; i += warpSize) {
      buf[threadIdx.x] += buf[i];
    }
    if (threadIdx.x < 16) {
      if (threadIdx.x + 16 < n_classes)
        buf[threadIdx.x] += buf[threadIdx.x + 16];
      if (threadIdx.x + 8 < n_classes)
        buf[threadIdx.x] += buf[threadIdx.x + 8];
      if (threadIdx.x + 4 < n_classes)
        buf[threadIdx.x] += buf[threadIdx.x + 4];
      if (threadIdx.x + 2 < n_classes)
        buf[threadIdx.x] += buf[threadIdx.x + 2];
      if (threadIdx.x + 1 < n_classes)
        buf[threadIdx.x] += buf[threadIdx.x + 1];
    }
  }
  if (threadIdx.x == 0)
    *gpu_res = buf[0];
}

REAL Gpu::ErrFctSoftmClassError(const int bsize, const int n_classes, REAL *gpu_class_out, REAL *gpu_class_target)
{
  REAL res;
  if (gpu_result==NULL) cudaMalloc(&gpu_result,sizeof(REAL));
  int n_threads = std::min(bsize, 512);
  int n_blocks = bsize / n_threads + ((bsize % n_threads) ? 1 : 0);
  int n_shared_bytes = n_threads * sizeof(REAL);
  KernelErrFctSoftmClassError2<<<n_blocks, n_threads, n_shared_bytes, Gpu::curStream>>>(
      bsize, n_classes, gpu_class_out, gpu_class_target, gpu_result);
  cudaMemcpyAsync(&res, gpu_result, sizeof(REAL), cudaMemcpyDeviceToHost, Gpu::curStream);
  cudaStreamSynchronize(Gpu::curStream);
  return res;
}

//-----------------------------------------------
// ErrFctSoftmCrossEntNgram::CalcGrad
//-----------------------------------------------
/**
 * @note This kernel need many block to compute the grad but also need to do a reduction.
 * The first block will do the reduction and compute the grad associated with it
 * and all the other will compute the grad for other words.
 */
__global__
void KernelErrFctSoftmCrossEntNgramCalcGrad(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_grad, REAL *gpu_target,
					    REAL *gpu_res)
{
  if (blockIdx.x == 0) {
    // the first block computes the error and grad for used words
    extern __shared__ REAL buf[];
    REAL err=0.0;
    for (int b=threadIdx.x; b<bsize; b+=blockDim.x) {
      unsigned int tidx=(uint) gpu_target[b];
      gpu_grad[b*odim + tidx] = (1.0f - gpu_grad[b*odim + tidx]);
      err += gpu_safelog(gpu_data_out[b*odim + tidx]);
    }
    buf[threadIdx.x] = err;
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int i=1; i<blockDim.x; i++)
        err += buf[i];
      *gpu_res=err;
    }
  }
  else
    // the next blocks computes the grad for all other words
    for (int b=blockIdx.x-1; b<bsize; b+=gridDim.x-1) {
      unsigned int tidx=(uint) gpu_target[b];
      for (int i=threadIdx.x; i<odim; i+=blockDim.x)
        if (tidx != (uint)i)
          gpu_grad[b*odim + i] *= -1.0f;
    }
}

void Gpu::ErrFctSoftmCrossEntNgramCalcGrad(const int bsize, const int odim, REAL *gpu_data_out,
                                         REAL *gpu_grad, REAL *gpu_target, REAL * gpu_res)
{
  cudaMemcpyAsync(gpu_grad, gpu_data_out, bsize*odim*sizeof(REAL), cudaMemcpyDeviceToDevice, Gpu::curStream);

  int nb_blocks = std::min(Gpu::curDevProps->maxGridSize[0], bsize + 1);
  int nb_threads = std::min(Gpu::curDevProps->maxThreadsDim[0], bsize);
  int n_shared_bytes = nb_threads * sizeof(REAL);
  KernelErrFctSoftmCrossEntNgramCalcGrad<<<nb_blocks, nb_threads, n_shared_bytes, Gpu::curStream>>>(
      bsize, odim, gpu_data_out, gpu_grad, gpu_target, gpu_res);

  cudaError_t err = cudaGetLastError();
  if(cudaSuccess != err){
    ErrorN("Error in Gpu::ErrFctSoftmCrossEntNgramCalcGrad: %s", cudaGetErrorString(err));
  }
}

//-----------------------------------------------
// ErrFctSoftmCrossEntNgram::CalcGradNull
//-----------------------------------------------
/**
 * @note This kernel need many block to compute the grad but also need to do a reduction.
 * The first block will do the reduction and compute the grad associated with it
 * and all the other will compute the grad for other words.
 */
__global__
void KernelErrFctSoftmCrossEntNgramCalcGradNull(const int bsize, const int odim,
     REAL *gpu_data_out, REAL *gpu_grad, REAL *gpu_target,
                                            REAL *gpu_res)
{
  if (blockIdx.x == 0) {
    // the first block computes the error and grad for non NULL words
    extern __shared__ REAL buf[];
    REAL err=0.0;
    for (int b=threadIdx.x; b<bsize; b+=blockDim.x) {
      //Do not cast or use unsigned for tidx. Otherwise, nvcc will transform the -1 to 0!
      //This is a difference compared to the GPU!
      int tidx = gpu_target[b];
      debug5(" -batch=%d target=%d -> output at %p is %f, update grad at %p\n", b, tidx, &(gpu_data_out[b*odim + tidx]), gpu_data_out[b*odim + tidx], &(gpu_grad[b*odim+tidx]));
      if (tidx != NULL_WORD) {
        gpu_grad[b*odim + tidx] = (1.0f - gpu_grad[b*odim + tidx]);
        err += gpu_safelog(gpu_data_out[b*odim + tidx]);
      }
    }
    buf[threadIdx.x] = err;
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int i=1; i<blockDim.x; i++)
        err += buf[i];
      *gpu_res=err;
    }
  }
  else
    // the next blocks computes the grad for all other words
    for (int b=blockIdx.x-1; b<bsize; b+=gridDim.x-1) {
      int tidx = gpu_target[b];
      for (int i=threadIdx.x; i<odim; i+=blockDim.x) {
        if (tidx == NULL_WORD)
          gpu_grad[b*odim + i] = 0;
        else if (tidx != i)
          gpu_grad[b*odim + i] *= -1.0f;
      }
    }
}

void Gpu::ErrFctSoftmCrossEntNgramCalcGradNull(const int bsize, const int odim, REAL *gpu_data_out,
                                         REAL *gpu_grad, REAL *gpu_target, REAL * gpu_res)
{
  cudaMemcpyAsync(gpu_grad, gpu_data_out, bsize*odim*sizeof(REAL), cudaMemcpyDeviceToDevice, Gpu::curStream);

  int nb_blocks = std::min(Gpu::curDevProps->maxGridSize[0], bsize + 1);
  int nb_threads = std::min(Gpu::curDevProps->maxThreadsDim[0], bsize);
  int n_shared_bytes = nb_threads * sizeof(REAL);
  KernelErrFctSoftmCrossEntNgramCalcGradNull<<<nb_blocks, nb_threads, n_shared_bytes, Gpu::curStream>>>(
      bsize, odim, gpu_data_out, gpu_grad, gpu_target, gpu_res);

  cudaError_t err = cudaGetLastError();
  if(cudaSuccess != err){
    ErrorN("Error in Gpu::ErrFctSoftmCrossEntNgramCalcGradNull: %s", cudaGetErrorString(err));
  }
}

//-----------------------------------------------
// ErrFctSoftmCrossEntNgram::CalcGradCumul
//-----------------------------------------------
/**
 * @note This kernel need many block to compute the grad but also need to do a reduction.
 * The first block will do the reduction and compute the grad associated with it
 * and all the other will compute the grad for other words.
 */
__global__
void KernelErrFctSoftmCrossEntNgramCalcGradCumul(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_grad, REAL *gpu_target,
					    REAL *gpu_res)
{
  if (blockIdx.x == 0) {
    // the first block computes the error and grad for used words
    extern __shared__ REAL buf[];
    REAL err=0.0;
    unsigned int tidx;

    for (int b=threadIdx.x ; b<bsize ; b+=blockDim.x) {
      tidx=(b*odim + (uint) gpu_target[b]);
      gpu_grad[tidx] = (1.0f - gpu_grad[tidx]);
      err += gpu_safelog(gpu_data_out[tidx]);
    }
    buf[threadIdx.x] = err;
    __syncthreads();
    if(threadIdx.x == 0) {
      for(int i=1 ; i<blockDim.x ; i++)
        err += buf[i];
      atomicAdd(gpu_res, err);
    }
  }
  else
    // the next blocks computes the grad for all other words
    for (int b=blockIdx.x-1; b<bsize; b+=gridDim.x-1) {
      unsigned int tidx = gpu_target[b];
      for (int i=threadIdx.x; i<odim; i+=blockDim.x)
        if (tidx != (uint)i)
          gpu_grad[b*odim + i] *= -1.0f;
    }
}


void Gpu::ErrFctSoftmCrossEntNgramCalcGradCumul(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_grad, REAL *gpu_target)
{
  if (gpu_result==NULL) cudaMalloc(&gpu_result,sizeof(REAL));

  cudaMemsetAsync(gpu_result, 0.0, sizeof(REAL), Gpu::curStream); //Each thread will atomicAdd into it.
  cudaMemcpyAsync(gpu_grad, gpu_data_out, bsize*odim*sizeof(REAL), cudaMemcpyDeviceToDevice, Gpu::curStream);
  int nb_blocks = std::min(Gpu::curDevProps->maxGridSize[0], bsize + 1);
  int n_threads = std::min(Gpu::curDevProps->maxThreadsDim[0], bsize);
  KernelErrFctSoftmCrossEntNgramCalcGradCumul<<<nb_blocks, n_threads, n_threads*sizeof(REAL), Gpu::curStream>>>(bsize, odim, gpu_data_out, gpu_grad, gpu_target, gpu_result);
  Error("Gpu::ErrFctSoftmCrossEntNgramCalcGradCumul not finished!");

  //REAL res;
  //cudaMemcpyAsync(&res, gpu_result, sizeof(REAL), cudaMemcpyDeviceToHost, Gpu::curStream);
  //cudaStreamSynchronize(Gpu::curStream);
  //return res;
}

//-----------------------------------------------
// ErrFctSoftmCrossEntNgramMulit::CalcGrad
//-----------------------------------------------
/**
 * @note This kernel need many block to compute the grad but also need to do a reduction.
 * The first part of blocks will do the reduction and compute the grad associated with it
 * and all the other will compute the grad for other words.
 */
__global__
void KernelErrFctSoftmCrossEntNgramMultiCalcGrad(const int bsize, const int dim, const int nb,
						 REAL *gpu_data_out, REAL *gpu_grad, REAL *gpu_target,
	                                         REAL *gpu_res)
{
  if (blockIdx.y == 0) {
    if (threadIdx.x < nb) {
      // the first part of blocks computes the error and grad for non NULL words
      extern __shared__ REAL buf[];
      REAL err=0.0;
      for (int b=blockIdx.x; b<bsize; b+=gridDim.x)
        for (int n=threadIdx.x; n<nb; n+=blockDim.x) {
          int tidx=(int) gpu_target[b*nb + n];
          if (tidx != NULL_WORD) {
            gpu_grad[(b*nb+n)*dim + tidx] = (1.0 - gpu_grad[(b*nb+n)*dim + tidx]);
            err += gpu_safelog(gpu_data_out[(b*nb+n)*dim + tidx]);
            debug6("grad ngram-multi:  b=%d, n=%d, tidx=%u, out=%f -> err=%e, grad@target=%e\n", b, n, tidx, gpu_data_out[(b*nb+n)*dim + tidx], err, gpu_grad[(b*nb+n)*dim + tidx]);
          }
          else {
            debug4("grad ngram-multi:  b=%d, n=%d, tidx=NULL, out=%f -> err=%e\n", b, n, gpu_data_out[(b*nb+n)*dim + tidx], err);
          }
        }
      buf[threadIdx.x] = err;
      __syncthreads();
      if (threadIdx.x == 0) {
        for (int i=1; (i<nb) && (i<blockDim.x); i++)
          err += buf[i];
        atomicAdd(gpu_res, err);
      }
    }
  }
  else if (threadIdx.x < dim)
    // the next blocks computes the grad for all other words
    for (int b=blockIdx.x; b<bsize; b+=gridDim.x)
      for (int n=(blockIdx.y-1); n<nb; n+=(gridDim.y-1)) {
        int tidx=(int) gpu_target[b*nb + n];
        for (int i=threadIdx.x; i<dim; i+=blockDim.x) {
          if (tidx == NULL_WORD)
            gpu_grad[(b*nb+n)*dim + i] = 0;
          else if (tidx != i)
            gpu_grad[(b*nb+n)*dim + i] *= -1.0;
        }
      }
}

REAL Gpu::ErrFctSoftmCrossEntNgramMultiCalcGrad(const int bsize, const int dim, const int nb,
                                              REAL *gpu_data_out, REAL *gpu_grad, REAL *gpu_target)
{
  if (gpu_result==NULL) cudaMalloc(&gpu_result, sizeof(REAL));

// same below
  int n=bsize*nb*dim;
  cudaMemcpyAsync(gpu_grad, gpu_data_out, n*sizeof(REAL),
             cudaMemcpyDeviceToDevice, Gpu::curStream);
  
  cudaMemsetAsync(gpu_result, 0.0, sizeof(REAL), Gpu::curStream);//Each block will atomicAdd into it.
 
  cudaError_t sts = cudaGetLastError();
  if (cudaSuccess != sts)
    Error("Error before KernelErrFctSoftmCrossEntNgramMultiCalcGrad");
  int nb_threads = std::min(std::max(nb, dim), Gpu::curDevProps->maxThreadsDim[0]);
  int n_shared_bytes = std::min(nb, nb_threads) * sizeof(REAL);
  dim3 nb_blocks(std::min( bsize, Gpu::curDevProps->maxGridSize[0]),
                 std::min(nb + 1, Gpu::curDevProps->maxGridSize[1]));
  KernelErrFctSoftmCrossEntNgramMultiCalcGrad<<<nb_blocks, nb_threads, n_shared_bytes, Gpu::curStream>>>(
    bsize, dim, nb, gpu_data_out, gpu_grad, gpu_target, gpu_result);
  sts = cudaGetLastError();
  if (cudaSuccess != sts) 
  {
    printf(cudaGetErrorString(sts));
    Error("KernelErrFctSoftmCrossEntNgramMultiCalcGrad cuda error: ");
  }
  REAL res;
  cudaMemcpyAsync(&res, gpu_result, sizeof(REAL), cudaMemcpyDeviceToHost, Gpu::curStream);
  cudaStreamSynchronize(Gpu::curStream);

  return res;
}


//-----------------------------------------------
// MachSoftmaxClass
//-----------------------------------------------
// Forw
/* This function performs the equivalent of various Gemv, with different sizes
   and offsets for each example in a minibatch. */
__global__ void KernelLinForwOffset(const int bsize, const int idim, const int odim,
                                    REAL* input, REAL* weights, REAL* bias, REAL* output,
                                    int* class_info)
{
  // Each block corresponds to one (or more) sub-vector of the output. Each thread
  // corresponds to one of its elements.
  // Axis x of the grid corresponds to the output rows: if sizes takes large values,
  // j will need to go beyond gridDim.x * blockDim.x
  // Axis y of the grid corresponds to the batch size.

  extern __shared__ REAL buf[];

  for (int i = blockIdx.y; i < bsize; i += gridDim.y) {
    int offset = class_info[2*i];
    int size = class_info[2*i+1];
    REAL* in_vec = input + i*idim;

    // Copy in_vec into shared memory, so all threads in this block can access it faster
    for (int k = threadIdx.x; k < idim; k += blockDim.x) {
      buf[k] = in_vec[k];
    }
    __syncthreads();

    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < size; j += blockDim.x * gridDim.x) {
      // Compute one (vector-vector) dot product
      REAL dot = bias[offset + j];
      REAL* w_vec = weights + offset + j;
      for (int k = 0; k < idim; k++) {
        dot += buf[k] * w_vec[k*odim];
      }
      output[i*odim + offset + j] = dot;
    }
  }
}

void Gpu::MachSoftmaxClassLinForw(const int bsize, const int idim, const int odim,
                                REAL* input, REAL* weights, REAL* bias, REAL* output,
                                int* class_info, const int max_size)
{
  debug4("bsize: %d, idim: %d, odim: %d, max_size: %d\n", bsize, idim, odim, max_size);
  int n_threads = std::min(Gpu::curDevProps->maxThreadsDim[0], max_size);
  int n_blocks_y = std::min(Gpu::curDevProps->maxGridSize[1], bsize);
  int n_blocks_x = std::min(Gpu::curDevProps->maxGridSize[0], max_size/n_threads + (max_size%n_threads==0?0:1));
  int n_shared_bytes = idim*sizeof(REAL);
  dim3 n_blocks(n_blocks_x, n_blocks_y);

  debug3("n_threads: %d, n_blocks: (%d, %d)\n", n_threads, n_blocks_x, n_blocks_y);
  KernelLinForwOffset<<<n_blocks, n_threads, n_shared_bytes, Gpu::curStream>>>(
      bsize, idim, odim, input, weights, bias, output, class_info);
  cudaError_t err = cudaGetLastError();
  if(cudaSuccess != err){
    printf("KernelLinForwOffset: n_blocks=(%d, %d), n_threads=%d, shared=%d bytes\n",
           n_blocks_x, n_blocks_y, n_threads, n_shared_bytes);
    Error(cudaGetErrorString(err));
  }
}

__global__ void KernelBatchedSoftmaxOffset(int M,
    const REAL * x, const int sx0, const int sx1,
    REAL * sm, const int sm_s0, const int sm_s1,
    int * offsets, const int offsets_s,
    int * sizes, const int sizes_s)
{
  extern __shared__ REAL buf[];
  for (int blockIDX = blockIdx.x; blockIDX < M; blockIDX += gridDim.x) {
    REAL sum = 0;
    int offset = offsets[blockIDX * offsets_s];
    int size = sizes[blockIDX * sizes_s];
#pragma unroll 16
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
      sum += exp(x[blockIDX * sx0 + (offset + i) * sx1]);
    }
    buf[threadIdx.x] = sum;
    __syncthreads();

    // This function trashes buf[1..warpsize], leaving the reduction result in buf[0].
    if (threadIdx.x < warpSize){
#pragma unroll 8
      for (int i = threadIdx.x + warpSize; i < blockDim.x && i < size; i += warpSize){
        buf[threadIdx.x] += buf[i];
      }
      if (threadIdx.x < 16){
        //reduce so that threadIdx.x 0 has the sum of everything
        if (threadIdx.x + 16 < size)
          buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+16];
        if (threadIdx.x + 8 < size)
          buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+8];
        if (threadIdx.x + 4 < size)
          buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+4];
        if (threadIdx.x + 2 < size)
          buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+2];
        if (threadIdx.x + 1 < size)
          buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+1];
      }
    }
    __syncthreads();
    REAL row_sum = buf[0];
#pragma unroll 16
    for (int i = threadIdx.x; i < size; i += blockDim.x){
      sm[blockIDX * sm_s0 + (offset + i) * sm_s1] = exp(x[blockIDX * sx0 + (offset + i) * sx1]) / row_sum;
    }
    __syncthreads();
  }
}

__global__ void KernelBatchedSoftmaxStableOffset(int M,
    const REAL * x, const int sx0, const int sx1,
    REAL * sm, const int sm_s0, const int sm_s1,
    int * offsets, const int offsets_s,
    int * sizes, const int sizes_s)
{
  extern __shared__ REAL buf[];
  for (int blockIDX = blockIdx.x; blockIDX < M; blockIDX += gridDim.x) {
    int offset = offsets[blockIDX * offsets_s];
    int size = sizes[blockIDX * sizes_s];
    REAL max_ = x[blockIDX * sx0 + (offset + threadIdx.x) * sx1];
    for (int i = threadIdx.x + blockDim.x; i < size; i += blockDim.x) {
      max_ = max(max_, x[blockIDX * sx0 + (offset + i) * sx1]);
    };
    buf[threadIdx.x] = max_;
    __syncthreads();

    // This function trashes buf[1..n_threads], leaving the reduction result in buf[0].
    // Find the max to stabilize the softmax
    if (threadIdx.x < warpSize)
    {
      for (int i = threadIdx.x + warpSize; i < blockDim.x && i < size; i += warpSize) {
        buf[threadIdx.x] = max(buf[threadIdx.x], buf[i]);
      }
      if (threadIdx.x < 16) {
        //reduce so that threadIdx.x 0 has the max of everything
        if (threadIdx.x + 16 < size)
          buf[threadIdx.x] = max(buf[threadIdx.x], buf[threadIdx.x+16]);
        if (threadIdx.x + 8 < size)
          buf[threadIdx.x] = max(buf[threadIdx.x], buf[threadIdx.x+8]);
        if (threadIdx.x + 4 < size)
          buf[threadIdx.x] = max(buf[threadIdx.x], buf[threadIdx.x+4]);
        if (threadIdx.x + 2 < size)
          buf[threadIdx.x] = max(buf[threadIdx.x], buf[threadIdx.x+2]);
        if (threadIdx.x + 1 < size)
          buf[threadIdx.x] = max(buf[threadIdx.x], buf[threadIdx.x+1]);
      }
    }
    __syncthreads();
    REAL row_max = buf[0];
    __syncthreads();

    REAL sum = 0;
#pragma unroll 16
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
      sum += exp(x[blockIDX * sx0 + (offset + i) * sx1] - row_max);
    }
    buf[threadIdx.x] = sum;
    __syncthreads();

    // This function trashes buf[1..warpsize], leaving the reduction result in buf[0].
    if (threadIdx.x < warpSize){
#pragma unroll 8
      for (int i = threadIdx.x + warpSize; i < blockDim.x && i < size; i += warpSize){
                buf[threadIdx.x] += buf[i];
      }
      if (threadIdx.x < 16) {
        //reduce so that threadIdx.x 0 has the sum of everything
        if (threadIdx.x + 16 < size)
          buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+16];
        if (threadIdx.x + 8 < size)
          buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+8];
        if (threadIdx.x + 4 < size)
          buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+4];
        if (threadIdx.x + 2 < size)
          buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+2];
        if (threadIdx.x + 1 < size)
          buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x+1];
      }
    }
    __syncthreads();
    REAL row_sum = buf[0];
#pragma unroll 16
    for (int i = threadIdx.x; i < size; i += blockDim.x){
      sm[blockIDX * sm_s0 + (offset + i) * sm_s1] = exp(x[blockIDX * sx0 + (offset + i) * sx1] - row_max) / row_sum;
    }
    __syncthreads();
  }
}

void Gpu::MachSoftmaxClassSoftmForw(const int bsize, const int odim, REAL* gpu_data_out,
                                  int* class_info, const int max_size, const int stable)
{
  int n_blocks = std::min(bsize, 32 * 1024);
  int n_threads = std::min(max_size, 512);
  int n_shared_bytes = n_threads * sizeof(REAL);
  if (bsize > 0) {
    if (stable) {
      KernelBatchedSoftmaxStableOffset<<<n_blocks, n_threads, n_shared_bytes, Gpu::curStream>>>(bsize,
          gpu_data_out, odim, 1,
          gpu_data_out, odim, 1,
          class_info, 2,
          class_info + 1, 2);
    }
    else {
      KernelBatchedSoftmaxOffset<<<n_blocks, n_threads, n_shared_bytes, Gpu::curStream>>>(bsize,
          gpu_data_out, odim, 1,
          gpu_data_out, odim, 1,
          class_info, 2,
          class_info + 1, 2);
      cudaError_t err = cudaGetLastError();
      if(cudaSuccess != err){
        printf("KernelBatchedSoftmaxOffset: n_blocks=%d, n_threads=%d, n_shared_bytes=%d odim=%d\n",
               n_blocks, n_threads, n_shared_bytes, odim);
        Error(cudaGetErrorString(err));
      }
    }
  }
}

__global__ void KernelBatchedSoftmCrossEntGradOffset(int M,
    const REAL* x, const int sx0, const int sx1,
    REAL* grad, const int sg0, const int sg1,
    REAL* target, const int st,
    int* offsets, const int so,
    int* sizes, const int ss,
    REAL* res)
{
  extern __shared__ REAL buf[];
  REAL err = 0.0f;
  for (int i = threadIdx.x; i < M; i += blockDim.x) {
    int offset = offsets[i * so];
    int size = sizes[i * ss];
    for (int j = 0; j < size; j++) {
      grad[i * sg0 + (offset + j) * sg1] = - x[i * sx0 + (offset + j) * sx1];
    }
    unsigned int tidx = (uint) target[i * st] - offset;
    grad[i * sg0 + (offset + tidx) * sg1] += 1.0f;
    err += gpu_safelog(x[i * sx0 + (offset + tidx) * sx1]);
  }
  buf[threadIdx.x] = err;
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int i = 1; i < blockDim.x; i++) {
      err += buf[i];
    }
    *res = err;
  }
}

void Gpu::ErrFctSoftmClassCrossEntNgramCalcGrad(const int bsize, const int odim,
    REAL* gpu_data_out, REAL* gpu_grad, REAL* gpu_target, int* class_info, REAL* gpu_res)
{
  int nb_threads = std::min(Gpu::curDevProps->maxThreadsDim[0], bsize);
  int n_shared_bytes = nb_threads * sizeof(REAL);
  KernelBatchedSoftmCrossEntGradOffset<<<1, nb_threads, n_shared_bytes, Gpu::curStream>>>(bsize,
      gpu_data_out, odim, 1,
      gpu_grad, odim, 1,
      gpu_target, 1,
      class_info, 2,
      class_info + 1, 2,
      gpu_res);

  cudaError_t err = cudaGetLastError();
  if(cudaSuccess != err){
    ErrorN("Error in Gpu::ErrFctSoftmClassCrossEntNgramCalcGrad: %s", cudaGetErrorString(err));
  }
}

__global__ void KernelLinGradInOffset(const int bsize, const int idim,
                                      REAL* grad_out, const int sgo0, const int sgo1,
                                      REAL* weights, const int sw0, const int sw1,
                                      REAL* grad_in, const int sgi0, const int sgi1,
                                      int* offsets, const int so,
                                      int* sizes, const int ss)
{
  /*
     Computes the a dot product (equivalent of gemv) on each row of grad_in,
     using a different part of grad_out and weights each time (determined
     from offsets and sizes).
     Each row of grad_in (index i) corresponds to one blockIdx.y.
     Columns of grad_in (lines of weights, index j) are split in groups
     indexed by blockIdx.x. Each group has blockDim.y indices, each index
     corresponds to a value of threadIdx.y.
     For each (i, j), a scalar (vector-vector) dot product is computed, over
     two vectors of length sizes[i], this sum is indexed by k. blockDim.x partial
     sums are computed in parallel and stored in buf[threadIdx.y][threadIdx.x],
     then a reduction steps computes the final dot product.
     We use threadIdx.x as the fast-moving index to maximize coalesced memory
     reads and writes.
  */
  extern __shared__ REAL buf[];
  for (int i = blockIdx.y; i < bsize; i += gridDim.y) {
    int offset = offsets[i * so];
    int size = sizes[i * ss];

    REAL* ograd_vec = grad_out + i * sgo0;
    REAL* buf_y = buf + blockDim.x * threadIdx.y;
    for (int j = blockDim.y * blockIdx.x + threadIdx.y; j < idim; j += gridDim.x * blockDim.y) {
      // Perform partially-summed dot product, stored in buf[]
      REAL* w_vec = weights + j * sw0 + offset * sw1;
      REAL dot = 0;
      for (int k = threadIdx.x; k < size; k += blockDim.x) {
        dot += ograd_vec[(offset + k) * sgo1] * w_vec[k * sw1];
      }
      buf_y[threadIdx.x] = dot;
      __syncthreads();

      // Perform the final summation into the first columns of buf[]
      // and accumulate the final result in grad_in
      if (threadIdx.x < 16 && threadIdx.x + 16 < size)
        buf_y[threadIdx.x] += buf_y[threadIdx.x + 16];
      if (threadIdx.x <  8 && threadIdx.x +  8 < size)
        buf_y[threadIdx.x] += buf_y[threadIdx.x + 8];
      if (threadIdx.x <  4 && threadIdx.x +  4 < size)
        buf_y[threadIdx.x] += buf_y[threadIdx.x + 4];
      if (threadIdx.x <  2 && threadIdx.x +  2 < size)
        buf_y[threadIdx.x] += buf_y[threadIdx.x + 2];
      if (threadIdx.x == 0)
        grad_in[i * sgi0 + j * sgi1] += buf_y[0] + buf_y[1];
    }
  }
}

void Gpu::MachSoftmaxClassLinGradIn(const int bsize, const int idim, const int odim,
                                  REAL* grad_out, REAL* weights, REAL* grad_in,
                                  int* class_info, const int max_size)
{
  int n_threads_x = Gpu::curDevProps->warpSize; // one warp
  int n_threads_y = std::min(Gpu::curDevProps->maxThreadsPerBlock / n_threads_x, Gpu::curDevProps->maxThreadsDim[1]); // Maximum possible
  int n_blocks_x = std::min(Gpu::curDevProps->maxGridSize[0], idim / n_threads_y + (idim%n_threads_y==0?0:1));
  int n_blocks_y = std::min(Gpu::curDevProps->maxGridSize[1], bsize);
  int n_shared_bytes = n_threads_x * n_threads_y * sizeof(REAL);
  dim3 n_threads(n_threads_x, n_threads_y);
  dim3 n_blocks(n_blocks_x, n_blocks_y);

  KernelLinGradInOffset<<<n_blocks, n_threads, n_shared_bytes, Gpu::curStream>>>(
      bsize, idim,
      grad_out, odim, 1,
      weights, odim, 1,
      grad_in, idim, 1,
      class_info, 2,
      class_info + 1, 2);

  cudaError_t err = cudaGetLastError();
  if(cudaSuccess != err){
    ErrorN("Error in Gpu::MachSoftmaxClassLinGrad: %s", cudaGetErrorString(err));
  }
}

__global__ void KernelLinGradUpdate(const int bsize, const int idim,
                                    REAL* input, const int si0, const int si1,
                                    REAL* grad_out, const int sg0, const int sg1,
                                    REAL* weights, const int sw0, const int sw1,
                                    REAL* bias, const int sb,
                                    int* offsets, const int so,
                                    int* sizes, const int ss,
                                    const REAL lrate, const REAL wdecay)
{
  /*
     Computes a series of rank-1 updates (equivalent of ger) on sub-matrices
     of weights. Also performs updates on bias directly proportional to
     the relevant sub-vectors of grad_out.
     Each row of grad_out and of input (index k) corresponds to one blockIdx.y.
     Rows of weights (columns of inputs, index i) split in groups indexed by
     blockIdx.x. Each group has blockDim.y indices, each index corresponds to a
     value of threadIdx.y.
     Columns of weights and grad_out (index j) are iterated over with blockDim.x
     parallel threads, indexed by threadIdx.x.

     Using blockDim.x == 1 warp seems to maximize speed.

     NOTE: Applying weight decay on the whole weight matrix would be too slow
     (in the order of +50% execution time), so we apply it in this kernel,
     only on the weights that were used for this minibatch.
     Since there is no atomic multiplication primitive, the value of weights we
     read before the update may have already been updated (by another example in
     the same minibatch), or not. It should not make a large difference.
  */


  for (int k = blockIdx.y; k < bsize; k += gridDim.y) {
    int offset = offsets[k * so];
    int size = sizes[k * ss];
    REAL* in_vec = input + k * si0;
    REAL* grad_vec = grad_out + k * sg0 + offset * sg1;

    for (int i = blockIdx.x * blockDim.y + threadIdx.y; i < idim; i += gridDim.x * blockDim.y) {
      REAL* w_vec = weights + i * sw0 + offset * sw1;
      for (int j = threadIdx.x; j < size; j += blockDim.x)
      {
         REAL update = lrate * (in_vec[i * si1] * grad_vec[j * sg1]
         // TODO: if wdecay > 0, this "+" sign should probably be a "-",
         // but this is the convention used in MachLin.cpp.
                                + wdecay * w_vec[j]);
         atomicAdd(w_vec + j * sw1, update);
      }

      // Block with i == 0 also updates the bias
      if (i == 0)
      {
        for (int j = threadIdx.x; j < size; j += blockDim.x)
          atomicAdd(bias + (offset + j) * sb, lrate * grad_vec[j * sg1]);
      }
    }
  }
}

void Gpu::MachSoftmaxClassLinGradUpdate(const int bsize, const int idim, const int odim,
                                      REAL* input, REAL* grad_out,
                                      REAL* weights, REAL* bias,
                                      int* class_info, const int max_size,
                                      const REAL lrate, const REAL wdecay)
{
  int n_threads_x = Gpu::curDevProps->warpSize; // one warp
  int n_threads_y = std::min(Gpu::curDevProps->maxThreadsPerBlock / n_threads_x, Gpu::curDevProps->maxThreadsDim[1]); // Maximum possible
  int n_blocks_x = std::min(Gpu::curDevProps->maxGridSize[0], idim / n_threads_y + (idim%n_threads_y==0?0:1));
  int n_blocks_y = std::min(Gpu::curDevProps->maxGridSize[1], bsize);
  dim3 n_threads(n_threads_x, n_threads_y);
  dim3 n_blocks(n_blocks_x, n_blocks_y);
  int n_shared_bytes = 0;

  KernelLinGradUpdate<<<n_blocks, n_threads, n_shared_bytes, Gpu::curStream>>>(
      bsize, idim,
      input, idim, 1,
      grad_out, odim, 1,
      weights, odim, 1,
      bias, 1,
      class_info, 2,
      class_info + 1, 2,
      lrate,
      wdecay);
}

//-----------------------------------------------
// Copy
//-----------------------------------------------
__global__
void KernelCopyVectorToMatrix(REAL * mat, REAL * vec, const int M, const int N)
{
  for(int b = blockIdx.x; b<M; b+=gridDim.x)
    for(int i = threadIdx.x; i<N; i+=blockDim.x)
      mat[b * N + i] = vec[i];
}

/*
 * This copy the vector on each line of the matrix.
 */
void Gpu::CopyVectorToMatrix(REAL * mat, REAL * vec, const int M, const int N)
{
  int nb_blocks = std::min(M, Gpu::curDevProps->maxGridSize[0]);
  int nb_threads = std::min(N, Gpu::curDevProps->maxThreadsDim[0]);
  debug4("Gpu::CopyVectorToMatrix(%p, %p %d %d)\n", mat, vec, M, N);
  KernelCopyVectorToMatrix<<<nb_blocks, nb_threads, 0, Gpu::curStream>>>(mat, vec, M, N);
  cudaError_t cuda_stat=cudaGetLastError();
  if (cuda_stat != cudaSuccess)
  { ErrorN("CUDA: ERROR %d in Gpu::CopyVectorToMatrix(%p, %p %d %d): %s\n",
           cuda_stat, mat, vec, M, N, cudaGetErrorString(cuda_stat));
  }
}

__global__
void KernelCopyMatrixToMatrixStrided(REAL * dst, REAL * src, const int M, const int N, const int row_stride)
{
  for(int b = blockIdx.x; b<M; b+=gridDim.x)
    for(int i = threadIdx.x; i<N; i+=blockDim.x)
      dst[b * row_stride + i] = src[b * N + i]; 
}

__global__
void KernelCopyMatrixStridedToMatrix(REAL * dst, REAL * src, const int M, const int N,
                                     const int row_stride_src)
{
  for(int b = blockIdx.x; b<M; b+=gridDim.x)
    for(int i = threadIdx.x; i<N; i+=blockDim.x)
      dst[b * N + i] = src[b * row_stride_src + i]; 
}

/*
 * This copy each line of a contiguous matrix to another matrix that is strided
 */
void Gpu::CopyMatrixToMatrixStrided(REAL * dst, REAL * src, const int M, const int N, const int row_stride)
{
  int nb_blocks = std::min(M, Gpu::curDevProps->maxGridSize[0]);
  int nb_threads = std::min(N, Gpu::curDevProps->maxThreadsDim[0]);
  KernelCopyMatrixToMatrixStrided<<<nb_blocks, nb_threads, 0, Gpu::curStream>>>(dst, src, M, N, row_stride);
  cudaError_t cuda_stat=cudaGetLastError();
  if (cuda_stat != cudaSuccess){
    ErrorN("CUDA: ERROR %d in Gpu::CopyMatrixToMatrixStrided: %s\n",
           cuda_stat, cudaGetErrorString(cuda_stat));
  }
}

/*
 * This copy each line of a strided matrix to another matrix that is contiguous
 */
void Gpu::CopyMatrixStridedToMatrix(REAL * dst, REAL * src, const int M, const int N, const int row_stride)
{
  int nb_blocks = std::min(M, Gpu::curDevProps->maxGridSize[0]);
  int nb_threads = std::min(N, Gpu::curDevProps->maxThreadsDim[0]);
  KernelCopyMatrixStridedToMatrix<<<nb_blocks, nb_threads, 0, Gpu::curStream>>>(dst, src, M, N, row_stride);
  cudaError_t cuda_stat=cudaGetLastError();
  if (cuda_stat != cudaSuccess){
    ErrorN("CUDA: ERROR %d in Gpu::CopyMatrixStridedToMatrix: %s\n",
           cuda_stat, cudaGetErrorString(cuda_stat));
  }
}

//-----------------------------------------------
// Multiple AXPY input row on one output row
//-----------------------------------------------

// Each block compute a fixed number of colums for all batch.
// This allow to have read coalesced and don't need atomic opartion.
__global__
void KernelBatchedAXPY(const int n, const REAL a, REAL * x, const int incx,
                       REAL * y, const int incy, const int nb_batch){
  for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
      idx += blockDim.x*gridDim.x){
    for(int b=0; b<nb_batch; b++){
      y[idx * incy] += a * x[b * n * incx + idx * incx];
    }
  }
}

void Gpu::BatchedAXPY(const int n, const REAL a, REAL * x, const int incx,
                    REAL * y, const int incy, const int nb_batch){
  int nb_threads = std::min(128, n);
  int nb_blocks = std::min(Gpu::curDevProps->maxGridSize[0], n/nb_threads+(n%nb_threads==0?0:1));
  nb_blocks = std::max(nb_blocks, 1);
  KernelBatchedAXPY<<<nb_blocks,nb_threads, 0, Gpu::curStream>>>(n, a, x, incx, y, incy, nb_batch); 
 
}


//-----------------------------------------------
// Element-wise exponential
//-----------------------------------------------
__global__ void KernelElemwiseExp(const int size, REAL *gpu_data_in, REAL *gpu_data_out) {
  for (int idx=blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
    gpu_data_out[idx] = exp(gpu_data_in[idx]);
  }
}

/*
 * Performs gpu_data_out[i] = exp(gpu_data_in[i]) for 0 <= i < size
 */
void Gpu::ElemwiseExp(const int size, REAL *gpu_data_in, REAL *gpu_data_out) {
  int nb_threads = std::min(size, Gpu::curDevProps->maxThreadsDim[0]);
  int nb_blocks = std::min(size/nb_threads + ((size%nb_threads ) == 0 ? 0 : 1), Gpu::curDevProps->maxGridSize[0]);
  KernelElemwiseExp<<<nb_blocks, nb_threads, 0, Gpu::curStream>>>(size, gpu_data_in, gpu_data_out);
}

//-----------------------------------------------
// Tanh and its gradient
//-----------------------------------------------
__global__ void KernelElemwiseTanh(const int size, REAL *gpu_data_in, REAL *gpu_data_out) {
  for (int idx=blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
    gpu_data_out[idx] = tanh(gpu_data_in[idx]);
  }
}

__global__ void KernelElemwiseTanhGrad(const int size, REAL *gpu_data_out, REAL *gpu_grad_out, REAL *gpu_grad_in) {
  for (int idx=blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
    REAL data_out = gpu_data_out[idx];
    gpu_grad_in[idx] = (1.0f - data_out * data_out) * gpu_grad_out[idx];
  }
}

/*
 * Performs gpu_data_out[i] = tanh(gpu_data_in[i]) for 0 <= i < size
 * where tanh(x) = sinh/cosh = (exp x - exp -x) / (exp x + exp -x)
 *               = (exp(2*x) - 1) / (exp(2*x) + 1)
 */
void Gpu::ElemwiseTanh(const int size, REAL *gpu_data_in, REAL *gpu_data_out) {
  int nb_threads = std::min(size, Gpu::curDevProps->maxThreadsDim[0]);
  int nb_blocks = std::min(size/nb_threads + ((size%nb_threads ) == 0 ? 0 : 1), Gpu::curDevProps->maxGridSize[0]);
  KernelElemwiseTanh<<<nb_blocks, nb_threads, 0, Gpu::curStream>>>(size, gpu_data_in, gpu_data_out);
}

/*
 * Performs gpu_grad_in[i] = (1 - gpu_data_out[i]**2) * gpu_grad_out[i]
 * for 0 <= i < size
 * which corresponds to the backpropagation of the gradient through tanh.
 */
void Gpu::ElemwiseTanhGrad(const int size, REAL *gpu_data_out, REAL* gpu_grad_out, REAL *gpu_grad_in) {
  int nb_threads = std::min(size, Gpu::curDevProps->maxThreadsDim[0]);
  int nb_blocks = std::min(size/nb_threads + ((size%nb_threads ) == 0 ? 0 : 1), Gpu::curDevProps->maxGridSize[0]);
  KernelElemwiseTanhGrad<<<nb_blocks, nb_threads, 0, Gpu::curStream>>>(size, gpu_data_out, gpu_grad_out, gpu_grad_in);
}

/*
 * set GPU memory to a value - equivalent to memset() on CPU
 */

__global__ void KernelMemSet(const int size, REAL *adr, REAL val) {
  for (int idx=blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
    adr[idx] = val;
  }
}

void Gpu::MemSet(REAL *adr, REAL val, int size) {
  int nb_threads = std::min(size, Gpu::curDevProps->maxThreadsDim[0]);
  int nb_blocks = std::min(size/nb_threads + ((size%nb_threads ) == 0 ? 0 : 1), Gpu::curDevProps->maxGridSize[0]);
  KernelMemSet<<<nb_blocks, nb_threads, 0, Gpu::curStream>>>(size, adr, val);
}

//-----------------------------------------------
// Helpers
//-----------------------------------------------

void Gpu::ResSet(REAL val) {
  cudaMemcpyAsync(gpu_result, &val, sizeof(REAL), cudaMemcpyHostToDevice, Gpu::curStream);
}

REAL Gpu::ResGet() {
  REAL val;
  cudaMemcpyAsync(&val, gpu_result, sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(Gpu::curStream);
  return val;
}
