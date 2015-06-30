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
 */

#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>
#ifdef CUDA
#include "Gpu.cuh"
#endif
#include "MachMulti.h"
#include "Tools.h"
#include "Toolsgz.h"
#include "WordList.h"
using namespace std;

int main (int argc, char* argv[])
{
  // verify parameters
  if (argc != 6) {
    string sProgName(argv[0]);
    size_t stEndPath = sProgName.find_last_of("/\\");
    if (string::npos != stEndPath)
      sProgName = sProgName.substr(stEndPath + 1);
    cout << sProgName << ' ' << cslm_version << " - A tool to parse n-grams with machine and record its output\n"
         << "Copyright (C) 2014 Holger Schwenk, University of Le Mans, France\n\n"
         << "Usage: " << sProgName << "  input_machine  input_file  vocabulary  machine_index  output_file\n"
         << "  Runs input machine with n-grams from input file and writes output of machine at given index to output file.\n\n"
         << "  machine_index is made of machine indexes (starting at 0) separated by a dot:\n"
         << "    2     means the third submachine of the input machine,\n"
         << "    2.0   means the first submachine of previous machine,\n"
         << "    2.0.1 means the second submachine of previous machine." << endl;
    return 2;
  }

  // verify characters in machine index
  for (char* sCur = argv[4] ; '\0' != *sCur ; sCur++)
    if ((('0' > *sCur) || ('9' < *sCur)) && ('.' != *sCur))
      ErrorN("bad character '%c' in machine index \"%s\"", *sCur, argv[4]);

  // read vocabulary
  cout << " - reading vocabulary '" << argv[3] << "'";
  WordList vocab;
  vocab.Read(argv[3]);
  cout << endl;

  // read input machine
  ifstream ifs;
  ifs.open(argv[1],ios::binary);
  cout << " - reading machine '" << argv[1] << "'" << endl;
  CHECK_FILE(ifs, argv[1]);
  Mach* mach_read = Mach::Read(ifs);
  ifs.close();
  if (NULL == mach_read)
    Error("no input machine available");
  mach_read->Info();
  cout << endl;
  if (NULL == dynamic_cast<MachMulti*>(mach_read))
    Error("the input machine must be a multi-machine");

  // search for submachine
  Mach* submach = mach_read;
  size_t stSubM;
  for (char* sCur = argv[4] ; '\0' != *sCur ; sCur++) {
    MachMulti* mach_multi = dynamic_cast<MachMulti*>(submach);
    int iRead = ((NULL != mach_multi) ?
                 sscanf(sCur, "%zu", &stSubM) : 0);
    if ((1 == iRead) && (stSubM < (size_t)mach_multi->MachGetNb()))
      submach = mach_multi->MachGet(stSubM);
    else
      submach = NULL;
    if (NULL == submach) {
      delete mach_read;
      if (NULL == mach_multi) {
        (*--sCur) = '\0';
        ErrorN("the machine at position \"%s\" is not a multi-machine", argv[4]);
      }
      else if (1 == iRead) {
        (*sCur) = '\0';
        ErrorN("bad machine index \"%s%zu\"", argv[4], stSubM);
      }
      else
        ErrorN("bad machine index \"%s\"", argv[4]);
    }
    sCur = strchr(sCur, '.');
    if (NULL == sCur)
      break;
  }

  // open input and output files
  cout << " - reading input from file '" << argv[2] << "'" << endl;
  inputfilestream inpf(argv[2]);
  if (!inpf) {
    delete mach_read;
    perror(argv[2]);
    Error();
  }
  cout << " - writing output to file '" << argv[5] << "'" << endl;
  outputfilestream outf(argv[5]);
  if (!outf) {
    inpf.close();
    delete mach_read;
    perror(argv[5]);
    Error();
  }

  // parse n-grams
  time_t t_beg, t_end;
  time(&t_beg);
  int idim = mach_read->GetIdim();
  int odim = submach->GetOdim();
  int bsize = mach_read->GetBsize();
  REAL input[idim * bsize * sizeof(REAL)];
#ifdef CUDA
  REAL* gpu_input = Gpu::Alloc(idim * bsize * sizeof(REAL), "machine input");
  mach_read->SetDataIn(gpu_input);
  REAL output[odim * bsize * sizeof(REAL)];
#else
  mach_read->SetDataIn(input);
  REAL* output;
#endif
  string str;
  WordList::WordIndex wi = vocab.GetIndex(WordList::WordUnknown);
  bool frequ_sort = vocab.FrequSort();
  REAL unk_idx = ((WordList::BadIndex != wi) ? (frequ_sort ? vocab.MapIndex(wi) : wi) : NULL_WORD);
  int ngrams = 0;
  int nb = 0;
  while (!inpf.eof()) {
    stringbuf sb;
    inpf.get(sb);
    if ((bsize > nb) && !inpf.eof()) {
      // read input
      inpf.get();
      istream istr(&sb);
      for (int i = 0 ; idim > i ; i++) {
        istr >> str;
        if (!istr)
          input[nb * idim + i] = NULL_WORD;
        else {
          wi = vocab.GetIndex(str.c_str());
          input[nb * idim + i] = ((WordList::BadIndex != wi) ? (frequ_sort ? vocab.MapIndex(wi) : wi) : unk_idx);
        }
      }
      nb++;
    }
    if ((bsize <= nb) || (inpf.eof() && (0 < nb))) {
      // run machine
#ifdef CUDA
      Gpu::MemcpyAsync(gpu_input, input, nb * idim * sizeof(REAL), cudaMemcpyHostToDevice);
#endif
      mach_read->Forw(nb);
#ifdef CUDA
      Gpu::MemcpyAsync(output, submach->GetDataOut(), nb * odim * sizeof(REAL), cudaMemcpyDeviceToHost);
      Gpu::StreamSynchronize();
#else
      output = submach->GetDataOut();
#endif

      // write output
      for (int n = 0 ; nb > n ; n++) {
        for (int i = 0 ; odim > i ; i++)
          outf << output[n * odim + i] << ' ';
        outf << endl;
        if (outf.fail()) {
          inpf.close();
          outf.close();
          delete mach_read;
#ifdef CUDA
          cudaFree(gpu_input);
#endif
          ErrorN("can't write to '%s'", argv[5]);
        }

        ngrams++;
        if ((ngrams % 1000) == 0)
          cout << " - " << ngrams << " n-grams processed\r" << flush;
      }
      nb = 0;
    }
  }
  inpf.close();
  outf.close();
  delete mach_read;
#ifdef CUDA
  cudaFree(gpu_input);
#endif
  time(&t_end);
  time_t dur = (t_end - t_beg);

  // display final statistics
  cout << " - " << ngrams << " n-grams processed\n"
       << " - total time: " << dur / 60 << "m" << dur % 60 << "s" << endl;

#ifdef CUDA
  Gpu::Unlock();
#endif
  return 0;
}
