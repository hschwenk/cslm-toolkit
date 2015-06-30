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

#include "ErrFctSoftmClassCrossEntNgram.h"

#ifdef BLAS_CUDA
#include "Gpu.cuh"
#endif



ErrFctSoftmClassCrossEntNgram::ErrFctSoftmClassCrossEntNgram(Mach &mach)
 : ErrFct(mach), grad_class(NULL)
{
#ifdef BLAS_CUDA
  err = NULL;
  host_err = NULL;
#endif
}

ErrFctSoftmClassCrossEntNgram::ErrFctSoftmClassCrossEntNgram(const ErrFctSoftmClassCrossEntNgram &efct)
 : ErrFct(efct), grad_class(NULL)
{
#ifdef BLAS_CUDA
  err = NULL;
  host_err = NULL;
#endif
}

ErrFctSoftmClassCrossEntNgram::~ErrFctSoftmClassCrossEntNgram()
{
#ifdef BLAS_CUDA
  if (grad_class)
    cudaFree(grad_class);
  if (host_err) {
    delete [] host_err;
    host_err = NULL;
  }
  if (err) {
    cudaFree(err);
    err = NULL;
  }
#else
  if (grad_class)
    delete [] grad_class;
#endif
}

void ErrFctSoftmClassCrossEntNgram::SetUp(MachSoftmaxClass* p_mach_class, WordList* p_wlist)
{
  wlist = p_wlist;
  mach_class = p_mach_class;

  // Get some information from wlist
  class_sizes = wlist->GetClassSizes();
  n_classes = class_sizes.size();

#ifdef BLAS_CUDA
  if (grad_class)
    cudaFree(grad_class);
  grad_class = Gpu::Alloc(n_classes*bsize, "class gradient in ErrFctSoftmClassCrossEntNgram");
  // allocate GPU memory to store errors (class and conditional NLLs)
  // and host memory to transfer it
  if (err)
    cudaFree(err);
  err = Gpu::Alloc(2, "sum of class NLL, then sum of conditional NLL");

  if (host_err)
    delete [] host_err;
  host_err = new REAL[2];
#else
  if (grad_class)
    delete [] grad_class;
  grad_class = new REAL[n_classes*bsize];
#endif

  // Build the output layer (softmax with classes)
  mach_class->SetUp(wlist);
}

//************
// this->class_output contains the probability of each of the n_classes class,
// for all examples in the minibatch.
// this->output contains, for each example in the minibatch, a collection of
// conditional probability vectors, one for each class: for each word i
// in that class c, P(w=i|c, h), the conditional probability that the next
// word (w) is i, given the class c and given the context (h).
// In this function, only the conditional probabilities for words belonging to the
// class of the target word, c(t), are correct (other are garbage).
REAL ErrFctSoftmClassCrossEntNgram::CalcValue(int eff_bsize)
{
  double err_value = 0.;
  if (eff_bsize <= 0)
    eff_bsize = bsize;

#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  // The first component of the cost is the NLL of the correct class
  err_value += Gpu::ErrFctSoftmCrossEntNgramCalcValue(eff_bsize, n_classes,
                                                    output_class, target_class);
  // Second part is the conditional NLL of the correct word in the class.
  err_value += Gpu::ErrFctSoftmCrossEntNgramCalcValue(eff_bsize, dim, output, target);
#else
  REAL *tcptr=target_class;
  REAL *ocptr=output_class;
  REAL *tptr = target;
  REAL *optr = output;
  for (int b=0; b<eff_bsize; b++) {
    // The first component of the cost is the NLL of the correct class
    err_value += safelog(ocptr[(int) *tcptr++]);
    // Second part is the conditional NLL of the correct word in the class.
    err_value += safelog(optr[(int) *tptr]);
    ocptr += n_classes;
    optr += dim;
    tptr++;
  }
#endif

  return (REAL) err_value;
}

#if 0 // not used any more use CalcValueBatch instead
REAL ErrFctSoftmClassCrossEntNgram::CalcValueNth(int idx)
{
#ifdef BLAS_CUDA
  Gpu::SetConfig(gpu_conf);
  Error("ErrFctSoftmClassCrossEntNgram::CalcValueNth not implemented yet on GPU");
  return 0.;
#else
  REAL *tcptr = target_class + idx;
  REAL *ocptr = output_class + idx*n_classes;
  REAL *tptr = target + idx;
  REAL *optr = output + idx*dim;
  // The first component of the cost is the NLL of the correct class
  // Second part is the conditional NLL of the correct word in the class.
  return safelog(ocptr[(int) *tcptr]) + safelog(optr[(int) *tptr]);
#endif
}
#endif

//*********************************************************************************r
// CalcValueBatch 

void ErrFctSoftmClassCrossEntNgram::CalcValueBatch(int eff_bsize, REAL *res)
{

  if (eff_bsize<=0) eff_bsize=bsize;

  Error("ErrFctSoftmClassCrossEntNgram::CalcValueBatch() not yet implemented");
}

//*********************************************************************************r
// CalcMax 

void ErrFctSoftmClassCrossEntNgram::CalcMax(int eff_bsize, REAL *res, int *pos)
{

  if (eff_bsize<=0) eff_bsize=bsize;

  Error("ErrFctSoftmClassCrossEntNgram::CalcMax() not yet implemented");
}


// Computes the derivative of the NLL on the softmax class output, and on the
// conditional softmax output corresponding to the target class.
// As in ErrFctSoftmCrossEntNgram::CalcGrad, we include the derivative through
// the softmax, to avoid wasteful computations.
REAL ErrFctSoftmClassCrossEntNgram::CalcGrad(int eff_bsize)
{
  double err_value = 0.;
  if (eff_bsize <= 0)
    eff_bsize = bsize;

#ifdef BLAS_CUDA
  // First part: gradient wrt the class output
  Gpu::SetConfig(gpu_conf);
  Gpu::ErrFctSoftmCrossEntNgramCalcGrad(eff_bsize, n_classes, output_class, grad_class,
                                      target_class, err);

  // Second part: gradient wrt the conditional NLL
  // We use offset and class size stored in target_class_info
    Gpu::ErrFctSoftmClassCrossEntNgramCalcGrad(eff_bsize, dim, output, grad, target,
                                           target_class_info, err + 1);
  Gpu::MemcpyAsync(host_err, err, 2*sizeof(REAL), cudaMemcpyDeviceToHost);
  Gpu::StreamSynchronize();
  Gpu::CheckError("ErrFctSoftmClassCrossEntNgram::CalcGrad");
  err_value = host_err[0] + host_err[1];
#else
  REAL *ocptr=output_class;
  REAL *gcptr=grad_class;
  REAL *tcptr=target_class;
  REAL *optr=output;
  REAL *gptr=grad;
  REAL *tptr=target;

  // initialize grad_class
  int n=eff_bsize*n_classes;
  REAL f1 = -1.0;
  memcpy(grad_class, output_class, n*sizeof(REAL));
  SCAL(&n, &f1, grad_class, &inc1);
  for (int b=0; b<eff_bsize; b++) {
    int offset = target_class_info[2*b];
    int clsize = target_class_info[2*b+1];

    if (*tptr<0.0) ErrorN("negative index %f at %d",*tptr,b);
    int tgt_idx = (int) *tptr;
    int tgt_class = (int) *tcptr;

    // compute grad_class
    gcptr[tgt_class] += 1.0;
    err_value += safelog(ocptr[tgt_class]);

    // update the part of grad that corresponds to the words in target class,
    // which is a segment starting at offset after the beginning
    // of the row, and of length clsize.
    memcpy(gptr+offset, optr+offset, clsize*sizeof(REAL));
    SCAL(&clsize, &f1, gptr+offset, &inc1);
    gptr[tgt_idx] += 1.0;
    err_value += safelog(optr[tgt_idx]);

    // Increment pointers for next example
    tcptr++;
    gcptr += n_classes;
    ocptr += n_classes;
    tptr++;
    gptr += dim;
    optr += dim;
  }
#endif
  return (REAL) err_value;
}

REAL ErrFctSoftmClassCrossEntNgram::CalcGradNull(int eff_bsize)
{
  Error("ErrFctSoftmClassCrossEntNgram::CalcGradNull not implemented yet");
  return 0.;
}

REAL ErrFctSoftmClassCrossEntNgram::CalcWordClassError(int eff_bsize)
{
  int err_value = 0;
  if (eff_bsize <= 0)
  eff_bsize = bsize;
#ifdef BLAS_CUDA
  return Gpu::ErrFctSoftmClassError(eff_bsize, n_classes, output_class, target_class);
#else
  REAL *tcptr=target_class;
  REAL *ocptr=output_class;
  for (int b=0; b<eff_bsize; b++) {
    // Find the maximum predicted class probability
    REAL max_oclass = ocptr[0];
    int argmax = 0;
    for (int i=1; i<n_classes; i++) {
      if (ocptr[i] > max_oclass) {
        argmax = i;
        max_oclass = ocptr[i];
      }
    }

    if ((int) *tcptr != argmax)
      err_value++;


    ocptr += n_classes;
    tcptr++;
  }
#endif
  return (REAL) err_value;
}
