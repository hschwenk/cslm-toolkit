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
 *
 *
 *
 * Machine that computes classification probabilities for words in a vocabulary
 * of size n, divided in c classes. The probabilities are computed as the probability
 * of a class c and the conditional probability of a word w given a class, given the
 * context h:
 * P(w|h) = P(c|h) * P(w|c,h)
 * Each of these probabilities is computed as a softmax:
 *   a_i = exp(a_i) / sum_k a_k
 * with a_k is the kth output of a linear machine
 *
 * There is one softmax for the probabilities of classes, and one for each of
 * the classes (that computes the probabilities of words in that class).
 *
 * This enables us to compute the log-likelihood of one word without having to
 * compute the probabilities for all words: we only need the probabilities of
 * all classes, and the conditional probability of all words in that class.
 */

#include "MachSoftmaxClass.h"
#include "MachSoftmaxStable.h"

#ifdef BLAS_CUDA
#include "Gpu.cuh"
#endif

MachSoftmaxClass::MachSoftmaxClass(const int p_idim, const int p_odim, const int p_bsize,
                                   const ulong p_nbfw, const ulong p_nbbw, const int p_stable)
  : MachLin(p_idim, p_odim, p_bsize, p_nbfw, p_nbbw), class_softm_mach(NULL), wlist(NULL),
    target_class_info(NULL), stable(p_stable)
{
  // class_softm_mach will be built when we know the number of classes,
  // i.e., when SetUp(wlist) will be called. -1 means uninitialized.
  n_classes = -1;
}

MachSoftmaxClass::MachSoftmaxClass(const MachSoftmaxClass &m)
  : MachLin(m), class_softm_mach(NULL), wlist(m.wlist), n_classes(m.n_classes), target_class_info(NULL), stable(m.stable)
{
  if (m.class_softm_mach != NULL) {
    if (stable) {
      MachSoftmaxStable* m_class_softm_mach = dynamic_cast<MachSoftmaxStable*>(m.class_softm_mach);
      if (m_class_softm_mach == NULL)
        Error("Could not convert m.class_softm_mach to a MachSoftmaxStable, but m.stable is true");
    }
    else {
      MachSoftmax* m_class_softm_mach = dynamic_cast<MachSoftmax*>(m.class_softm_mach);
      if (m_class_softm_mach == NULL)
        Error("Could not convert m.class_softm_mach to a MachSoftmax, but m.stable is false");
    }
    class_softm_mach = m.class_softm_mach->Clone();
  }
}

MachSoftmaxClass::~MachSoftmaxClass()
{
  if (class_softm_mach) {
    delete class_softm_mach;
    class_softm_mach = NULL;
  }
}

ulong MachSoftmaxClass::GetNbParams()
{
  ulong nb_params = MachLin::GetNbParams();
  if (class_softm_mach)
    nb_params += class_softm_mach->GetNbParams();
  return nb_params;
}

void MachSoftmaxClass::Info(bool detailed, char *txt)
{
  if (detailed) {
    printf("MachSoftmaxClass::Info\n");
    Mach::Info(true, txt);
  }
  printf("%sMachSoftmaxClass %d-%d[%d], bs=%d, passes=%lu/%lu", txt, idim, odim, n_classes, bsize, nb_forw, nb_backw);
  if (lr_coeff != 1.0) printf(", lrate-coeff=%.2f", lr_coeff);
#ifdef BLAS_CUDA
  printf(", on GPU %d", Gpu::GetCudaDevice(Gpu::GetDevice(gpu_conf)));
#endif
  tm.disp(", ");
  tm.newline();

  char ntxt[256];
  snprintf(ntxt, 256, "%s  ", txt);
  if (class_softm_mach)
    class_softm_mach->Info(detailed, ntxt);
  else
    printf("%s<un-initialized MachSoftmax>\n", ntxt);
}

bool MachSoftmaxClass::CopyParams(Mach* mach)
{
  MachSoftmaxClass* mach_softmax_class = static_cast<MachSoftmaxClass*>(mach);
  if (Mach::CopyParams(mach))
  {
    if ((class_softm_mach == NULL) && (mach_softmax_class->class_softm_mach != NULL))
      // TODO: should we create it instead?
      return false;

    if ((class_softm_mach != NULL) && (mach_softmax_class->class_softm_mach == NULL))
      return false;

    if ((class_softm_mach != NULL) && (mach_softmax_class->class_softm_mach != NULL))
      // TODO: accept copying from a MachSoftmax into a MachSoftmaxStable and vice-versa
      if (!class_softm_mach->CopyParams(mach_softmax_class->class_softm_mach))
        return false;

    wlist = mach_softmax_class->wlist;
    n_classes = mach_softmax_class->n_classes;
    return true;
  }
  return false;
}

void MachSoftmaxClass::SetUp(WordList* p_wlist)
{
  wlist = p_wlist;
  vector<int> class_sizes = wlist->GetClassSizes();
  n_classes = class_sizes.size();
  max_class_size = 0;
  for (int i=0; i<n_classes; i++) {
    if (class_sizes[i] > max_class_size)
      max_class_size = class_sizes[i];
  }
  if (class_softm_mach) {
    // Check that the existing Mach has the right size
    int mach_nclasses = class_softm_mach->GetOdim();
    if (mach_nclasses != n_classes)
      ErrorN("In MachSoftmaxClass, class_softm_mach has output dim %d, incompatible with the number of classes of the WordList (%d)",
             mach_nclasses, n_classes);
  }
  else {
    if (stable) {
      MachSoftmaxStable* m = new MachSoftmaxStable(idim, n_classes, bsize, nb_forw, nb_backw);
      class_softm_mach = m;
      //TODO options for initialization
      //m->WeightsConst(0.);
      //m->BiasConst(0.);
      m->WeightsRandom(0.1);
      m->BiasRandom(0.1);
    }
    else {
      MachSoftmax* m = new MachSoftmax(idim, n_classes, bsize, nb_forw, nb_backw);
      class_softm_mach = m;
      //TODO options for initialization
      //m->WeightsConst(0.);
      //m->BiasConst(0.);
      m->WeightsRandom(0.1);
      m->BiasRandom(0.1);
    }
  }
  class_softm_mach->SetDataIn(data_in);
}

void MachSoftmaxClass::Forw(int eff_bsize, bool in_train)
{
  tm.start();
  if (eff_bsize <= 0)
    eff_bsize = bsize;

  // First, compute the classification probabilities
  class_softm_mach->Forw(eff_bsize,in_train);

  // TODO: if we want all (scaled) probabilities,
  // make that option available with a switch.
  // Here, we do the forward propagation of only the words corresponding
  // to the target class, as read from target_class_info
  if (target_class_info == NULL)
    Error("Cannot compute target probabilities without information on the actual target class");

#ifdef BLAS_CUDA
  // Use kernels to compute data_out directly and more efficiently
  Gpu::MachSoftmaxClassLinForw(eff_bsize, idim, odim, data_in, w, b, data_out,
                             target_class_info, max_class_size);
  Gpu::MachSoftmaxClassSoftmForw(eff_bsize, odim, data_out, target_class_info,
                               max_class_size, stable);
#else
  int one_i = 1;
  REAL one_f = 1.f;
  char transN = 'N';
  for (int k=0; k<eff_bsize; k++) {
    // Compute the affine part directly in data_out
    int offset = target_class_info[2*k];
    int clsize = target_class_info[2*k+1];
    REAL* sub_b = b + offset;
    REAL* sub_w = w + offset;
    REAL* data_in_k = data_in + k*idim;
    REAL* sub_dest = data_out + k*odim + offset;
    memcpy(sub_dest, sub_b, clsize*sizeof(REAL));
    GEMV(&transN, &clsize, &idim, &one_f, sub_w, &odim, data_in_k, &one_i, &one_f, sub_dest, &one_i);

    if (stable) {
      // Get the maximum value of sub_dest
      REAL max = sub_dest[0];
      for (int i=1; i<clsize; i++) {
        REAL x = sub_dest[i];
        if (x > max)
          max = x;
      }
      // Compute exp(x - max) inplace for all x in sub_dest, and their sum
      REAL sum_exp = 0.;
      for (int i=0; i<clsize; i++) {
        REAL exp_x = exp(sub_dest[i] - max);
        sum_exp += exp_x;
        sub_dest[i] = exp_x;
      }
      // Normalize sub_dest, dividing all values by sum_exp
      for (int i=0; i<clsize; i++) {
        sub_dest[i] /= sum_exp;
      }
    }
    else {
      // Apply exp on the output
      VEXP(&clsize, sub_dest);
      // compute normalization constant, the exp is positive, so we can use the sum of abs
      REAL norm = 1.0f / ASUM(&clsize, sub_dest, &one_i);
      // normalize
      SCAL(&clsize, &norm, sub_dest, &one_i);
    }
  }
#endif
  nb_forw += eff_bsize;
  tm.stop();
}

void MachSoftmaxClass::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  // As in MachSoftmax::Backw, we rely here on the error function
  // (ErrFctSoftmClassCrossEntNgram) to compute the gradient of the cost
  // wrt the linear activation (output of the affine part), NOT wrt
  // the output of the softmax.

  //Ignore weight decay for this, forward it to class_softm_mach
  //if (wdecay > 0.)
  //  Error("No weight decay implemented in MachSoftmaxClass, as the output weight matrix is too big to do so efficiently");

  if (drop_out > 0.)
    Error("No drop-out implemented in MachSoftmaxClass");

  if (eff_bsize <= 0)
    eff_bsize = bsize;

  if (!grad_out)
    Error("MachSoftmaxClass::Backw(): grad_out is not set");

  tm.start();

  // scale by block size !
  REAL lrate_bs = lr_coeff * lrate / sqrt(GetBsize());

  // First, initialize the input gradient by the one computed for the
  // class predictor, and update its parameters
  class_softm_mach->Backw(lrate_bs, wdecay, eff_bsize);
#ifdef BLAS_CUDA
  Gpu::MemcpyAsync(grad_in, class_softm_mach->GetGradIn(), eff_bsize*idim*sizeof(REAL),
                  cudaMemcpyDeviceToDevice);

  Gpu::MachSoftmaxClassLinGradIn(eff_bsize, idim, odim, grad_out, w, grad_in,
                               target_class_info, max_class_size);

  // TODO use lrate here? Each example in the batch will usually update different
  // parts of the weight matrix.
  Gpu::MachSoftmaxClassLinGradUpdate(eff_bsize, idim, odim,
                                   data_in, grad_out, w, b, target_class_info,
                                   max_class_size, lrate_bs, wdecay);

#else
  memcpy(grad_in, class_softm_mach->GetGradIn(), eff_bsize*idim*sizeof(REAL));

  int one_i = 1;
  REAL one_f = 1.f;
  char transT = 'T';
  REAL eps = 1 + lrate_bs * wdecay;

  // Then, for each example k in the minibatch, accumulate into grad_in:
  // grad_in[k] += sub_w ^T         * sub_grad_out[k]
  // (idim)       (clsize x idim)^T   (clsize)
  for (int k=0; k<eff_bsize; k++) {
    int offset = target_class_info[2*k];
    int clsize = target_class_info[2*k+1];
    REAL* sub_w = w + offset;
    REAL* sub_grad_out = grad_out + k*odim + offset;
    REAL* grad_in_k = grad_in + k*idim;
    GEMV(&transT, &clsize, &idim, &one_f, sub_w, &odim, sub_grad_out, &one_i, &one_f, grad_in_k, &one_i);
  }

  if (update) {
    // TODO use lrate here? Each example in the batch will usually update different
    // parts of the weight matrix.
    // Finally, for each example, update the parameters of the conditional predictor.
    // (Weight decay not implemented.)
    // sub_b += lrate * sub_grad_out[k]
    // sub_w          += lrate * sub_grad_out[k] * input[k]'
    // (clsize x idim)           (clsize)          (idim)
    // This is implemented by BLAS's ger routine:
    //   ger(m, n, alpha, x, incx, y, incy, A, lda)
    //   A += alpha * x * y^T
    for (int k=0; k<eff_bsize; k++) {
      int offset = target_class_info[2*k];
      int clsize = target_class_info[2*k+1];
      REAL* data_in_k = data_in + k*idim;
      REAL* sub_grad_out_k = grad_out + k*odim + offset;
      REAL* sub_b = b + offset;
      REAL* sub_w = w + offset;
      AXPY(&clsize, &lrate_bs, sub_grad_out_k, &one_i, sub_b, &one_i);
      GER(&clsize, &idim, &lrate_bs, sub_grad_out_k, &one_i, data_in_k, &one_i, sub_w, &odim);
      if (wdecay != 0)
        SCAL(&clsize, &eps, sub_w, &one_i);
    }
  }
#endif

  nb_backw += eff_bsize;
  tm.stop();
}

REAL* MachSoftmaxClass::GetDataOutClass()
{
  if (class_softm_mach)
    return class_softm_mach->GetDataOut();
  else
    return NULL;
}

void MachSoftmaxClass::SetGradOutClass(REAL* data)
{
  if (class_softm_mach)
    class_softm_mach->SetGradOut(data);
  else
    Error("MachSoftmaxClass is not initialized");
}

void MachSoftmaxClass::SetTargetInfo(int* info)
{
  // info contains the offset and lengths of the class in the vocabulary, so they
  // are needed if we want to only compute outputs for words in that class.
  target_class_info = info;
}

void MachSoftmaxClass::WriteData(ostream &outf)
{
  MachLin::WriteData(outf);
  outf.write((char*) &n_classes, sizeof(int));
  if (n_classes > 0) {
    // That should mean class_softm_mach is initialized, so we save it as well
    if (class_softm_mach == NULL)
      ErrorN("In MachSoftmaxClass::WriteData, n_classes (%d) > 0, but class_softm_mach is NULL", n_classes);
    class_softm_mach->Write(outf);
  }
}

void MachSoftmaxClass::ReadData(istream &inpf, size_t s, int bs)
{
  debug0("* read data in MachSoftmaxClass");
  MachLin::ReadData(inpf, s, bs);
  inpf.read((char*) &n_classes, sizeof(int));

  if (n_classes > 0) {
    // class_softm_mach was saved, we should try to load it
    if (class_softm_mach) {
      Error("Trying to read parameters for the class predictor from file, but one is already allocated");
    }
    Mach* sub_mach = Mach::Read(inpf, bs);
    if (stable) {
      class_softm_mach = dynamic_cast<MachSoftmaxStable*>(sub_mach);
      if (class_softm_mach == NULL)
      ErrorN("In MachSoftmaxClass::ReadData, n_classes(%d) > 0, and stable == true, but we could not load class_softm_mach as a MachSoftmaxStable", n_classes);
    }
    else {
      class_softm_mach = dynamic_cast<MachSoftmax*>(sub_mach);
      if (class_softm_mach == NULL)
      ErrorN("In MachSoftmaxClass::ReadData, n_classes(%d) > 0, and stable == false, but we could not load class_softm_mach as a MachSoftmax", n_classes);
    }
  }
}
