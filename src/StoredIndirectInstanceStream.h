/* Open source system for classification learning from very large data
** Class for an input stream where instances are stored as pointers to stored instances.
** Instances are stored in core and selected uniformly at random form those supplied as candidates
**
** Copyright (C) 2012 Geoffrey I Webb
**
** This program is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with this program. If not, see <http://www.gnu.org/licenses/>.
**
** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
*/
#pragma once
#include "IndirectInstanceStream.h"
#include "instanceStreamFilter.h"
#include "mtrand.h"

#include <vector>

class StoredIndirectInstanceStream : public IndirectInstanceStream
{
public:
  StoredIndirectInstanceStream();
  StoredIndirectInstanceStream(AddressableInstanceStream &source);   ///< load a sample from an instance stream
  ~StoredIndirectInstanceStream(void);

  void setSource(InstanceStream &source);                     ///< set the source for the sample and load it
  void setSourceWithoutLoading(InstanceStream &source);       ///< set the source for the sample without loading it
  void add(instance* inst);                                   ///< add the instance to the sample

private:
  std::vector<instance*> store_;  ///< the stored sample.  The sample is stored as pointers rather than as copies.
};
