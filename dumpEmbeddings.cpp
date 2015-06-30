/*
 * This file is part of the continuous space language and translation model toolkit
 * for statistical machine translation and large vocabulary speech recognition.
 *
 * Copyright 2015, Yannick Esteve, LIUM, University of Le Mans, France
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
 */

#include <cstdio>
#include <iostream>
#include <string>
#include "MachMulti.h"
#include "MachLin.h"
#include "MachTab.h"
#include "WordList.h"

using namespace std;

int main (int argc, char* argv[])
{
  // verify parameters
  if (argc != 4) {
    string sProgName(argv[0]);
    size_t stEndPath = sProgName.find_last_of("/\\");
    cerr << "Usage: " << ((stEndPath != string::npos) ? sProgName.substr(stEndPath + 1) : sProgName)
         << "  input_machine_file word_listi word_embeddings.txt\n"
         << "  Dump word embeddings provided by the projection layer machine.\n\n"
         << endl;
    return EXIT_FAILURE;
  }

  // read input machine
  ifstream ifs;
  ifs.open(argv[1],ios::binary);
  cerr << "Read machine from: " << argv[1] << endl;
  CHECK_FILE(ifs, argv[1]);
  Mach* mach_read = Mach::Read(ifs);
  ifs.close();

  if (NULL == mach_read)
    Error("no input machine available");
  mach_read->Info(true);
  cerr << endl;


  MachMulti* mach_multi = dynamic_cast<MachMulti*>(mach_read);
  if (mach_multi ==  NULL)
    Error("the input machine must be a multi-machine");

  MachMulti* submach_multi = dynamic_cast<MachMulti*>(mach_multi->MachGet(0)); // extraction of the projection layer
  if (submach_multi == NULL)
    Error("the projection layer machine must be contained in a multi-machine");

  MachTab* mach_write = dynamic_cast<MachTab*>(submach_multi->MachGet(0)); // extraction of the first machine from the projection layer 
  if (mach_write ==  NULL)
    Error("machines of the projection layer multi-machine must be table machines ");
  
  // read word list
  WordList wlist;
  char *wl_fname = argv[2];
  bool stable_sort=true;	// use stable sort (default=true), set to false for compatibility with CSLM <= V3.0

  cerr << " - reading word list from file " << wl_fname;
  wlist.SetSortBehavior(stable_sort);
  WordList::WordIndex voc_size = wlist.Read(wl_fname);
  cerr << endl;
  cerr << voc_size << " words in the vocabulary (word list)" << endl;

  int idim=0, odim=0;
 	REAL *myTable= mach_write->WeightTable(idim, odim);
  cerr <<"idim=number of words="<<idim<<", odim=embedding_ size="<<odim<<endl;
 
  ofstream ofs;
  ofs.open(argv[3]);
  int i=0, j=0;
  for (i=0; i<idim; i++) {
    ofs << wlist.GetWordInfoMapped(i).word <<" ";
    for (j=0; j<odim; j++) {
      ofs << myTable[i*odim + j] << " ";
    }
    ofs <<endl;
  }
  ofs.close();

  delete mach_read;

  GpuUnlock();
  return EXIT_SUCCESS;
}
