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

#include "WordList.h"
#include <iostream>
#include <string>
using namespace std;

int main (int argc, char *argv[])
{
  if (3 != argc) {
    string sProgName(argv[0]);
    size_t stEndPath = sProgName.find_last_of("/\\");
    cout << "Usage: " << ((stEndPath != string::npos) ? sProgName.substr(stEndPath + 1) : sProgName)
         << "  input_word_list  output_word_list\n"
         << "  Creates a new word list file compatible with stable and unstable sort." << endl;
    return 2;
  }

  WordList wl;

  /* open word list file with unstable sort */
  cout << "Reading " << argv[1] << flush;
  wl.SetSortBehavior(false);
  WordList::WordIndex n_words = wl.Read(argv[1]);
  cout << ", done." << endl;

  /* modify word counts to keep order with stable sort */
  cout << "Modifying word counts" << flush;
  WordList::const_iterator end = wl.End();
  for (WordList::const_iterator ci = wl.Begin() ; ci != end ; ci++)
    wl.GetWordInfo(ci->id).n = n_words--;
  cout << ", done." << endl;

  /* save new word list file */
  cout << "Writing " << argv[2] << flush;
  wl.Write(argv[2], 3);
  cout << ", done." << endl;

  return 0;
}
