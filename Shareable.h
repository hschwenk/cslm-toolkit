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

#ifndef _Shareable_h
#define _Shareable_h

class Shareable {
    protected:
    public:
	int iShareId;
	bool bExternal;
	Shareable();
	Shareable(bool=false, int=-1);
	int GetShareId() { return iShareId; }
	void SetShareId(int shareid){ iShareId = shareid; }
	bool HasExternalData() { return bExternal; }
	void SetExternalData(bool ext) { bExternal = ext; }

};

#endif


