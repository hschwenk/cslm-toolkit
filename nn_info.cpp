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

using namespace std;
#include <iostream>
#include "Mach.h"

int main (int argc, char *argv[])
{
  ifstream ifs;
  Mach *m;

  for (int i=1; i<argc; i++) {
    ifs.open(argv[i],ios::binary);
    CHECK_FILE(ifs,argv[i]);
    cout << endl << "Information on machine: " << argv[i] << endl;
    m = Mach::Read(ifs);
    cout << "Using file version " << Mach::GetFileId() << endl;
    m->Info();
    ifs.close();
    delete m;
  }

  GpuUnlock();
  return 0;
}
