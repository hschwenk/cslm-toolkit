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
 */

/*
 * from http://yann.lecun.com/exdb/mnist/
 
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label

The labels values are 0 to 9.

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
*/

#ifndef _DataMnist_h
#define _DataMnist_h

#include <iostream>
#include <fstream>

#include "DataFile.h"

extern const char* DATA_FILE_MNIST;

class DataMnist : public DataFile
{
protected:
  int dfd;		// file descriptor for data
  int lfd; 		// file descriptor for labels
  char *cl_fname;	// file name of classes
  REAL	tgt0, tgt1;	// low and high values of targets (e.g. -0.6/0.6 for tanh; 0/1 for softmax, ...)
  unsigned char *ubuf;	// input buffer
  uint read_iswap(int);	// read integer from file and swap bytes
public:
  DataMnist(char *, ifstream &ifs, int, const string&, int, string&, int, DataMnist* =NULL);	// optional object to initialize when adding factors
  virtual ~DataMnist();
  virtual void Rewind();
  virtual bool Next();
};

#endif
