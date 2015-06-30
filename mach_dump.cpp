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
#include <iostream>
#include <string>
#include "MachMulti.h"
using namespace std;

int main (int argc, char* argv[])
{
  // verify parameters
  if (argc != 4) {
    string sProgName(argv[0]);
    size_t stEndPath = sProgName.find_last_of("/\\");
    cout << "Usage: " << ((stEndPath != string::npos) ? sProgName.substr(stEndPath + 1) : sProgName)
         << "  input_machine_file  machine_index  output_machine_file\n"
         << "  Extracts a machine from input file at given machine index and write it in output file.\n\n"
         << "  machine_index is made of machine indexes (starting at 0) separated by a dot:\n"
         << "    2     means the third submachine of the input machine,\n"
         << "    2.0   means the first submachine of previous machine,\n"
         << "    2.0.1 means the second submachine of previous machine." << endl;
    return 2;
  }

  // verify characters in machine index
  for (char* sCur = argv[2] ; '\0' != *sCur ; sCur++)
    if ((('0' > *sCur) || ('9' < *sCur)) && ('.' != *sCur))
      ErrorN("bad character '%c' in machine index \"%s\"", *sCur, argv[2]);

  // read input machine
  ifstream ifs;
  ifs.open(argv[1],ios::binary);
  cout << "Read machine from: " << argv[1] << endl;
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
  Mach* mach_write = mach_read;
  size_t stSubM;
  for (char* sCur = argv[2] ; '\0' != *sCur ; sCur++) {
    MachMulti* mach_multi = dynamic_cast<MachMulti*>(mach_write);
    int iRead = ((NULL != mach_multi) ?
                 sscanf(sCur, "%zu", &stSubM) : 0);
    if ((1 == iRead) && (stSubM < (size_t)mach_multi->MachGetNb()))
      mach_write = mach_multi->MachGet(stSubM);
    else
      mach_write = NULL;
    if (NULL == mach_write) {
      delete mach_read;
      if (NULL == mach_multi) {
        (*--sCur) = '\0';
        ErrorN("the machine at position \"%s\" is not a multi-machine", argv[2]);
      }
      else if (1 == iRead) {
        (*sCur) = '\0';
        ErrorN("bad machine index \"%s%zu\"", argv[2], stSubM);
      }
      else
        ErrorN("bad machine index \"%s\"", argv[2]);
    }
    sCur = strchr(sCur, '.');
    if (NULL == sCur)
      break;
  }

  // write submachine
  ofstream ofs;
  ofs.open(argv[3],ios::binary);
  cout << "Write dumped machine to: " << argv[3] << endl;
  CHECK_FILE(ofs, argv[3]);
  mach_write->Info();
  mach_write->Write(ofs);
  ofs.close();
  delete mach_read;

  GpuUnlock();
  return 0;
}
