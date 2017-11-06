/* Open source system for classification learning from very large data
** Class for an input stream of randomly sampled instances
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
#include "StoredIndirectInstanceStream.h"
#include <algorithm>
#include <assert.h>

/// create the instance stream without any initialisation.  Must call setSource or setSourceWithoutLoading before use
StoredIndirectInstanceStream::StoredIndirectInstanceStream()
{
}

/// load a sample from an instance stream
StoredIndirectInstanceStream::StoredIndirectInstanceStream(AddressableInstanceStream &source)
{ setSource(source);
}

StoredIndirectInstanceStream::~StoredIndirectInstanceStream(void)
{
}

void StoredIndirectInstanceStream::setSource(InstanceStream &source) {
  assert(dynamic_cast<AddressableInstanceStream*>(&source) != NULL);  // we really require an AddressableInstanceStream

  source_ = &source;
  metaData_ = source.getMetaData();

  store_.clear();
  source.rewind();

  while (source.advance()) {
    store_.push_back(dynamic_cast<AddressableInstanceStream*>(&source)->current());
  }

  start_ = &store_[0];
  end_ = start_ + store_.size();
  
  rewind();
}

/// set the source for the sample without loading it
void StoredIndirectInstanceStream::setSourceWithoutLoading(InstanceStream &source) {
  assert(dynamic_cast<AddressableInstanceStream*>(&source) != NULL);  // we really require an AddressableInstanceStream

  source_ = &source;
  metaData_ = source.getMetaData();
  store_.clear();
  start_ = NULL;
  end_ = NULL;
  rewind();
}

/// add the instance to the sample
void StoredIndirectInstanceStream::add(instance* inst) {
  store_.push_back(inst);
  start_ = &store_[0];
  end_ = start_ + store_.size();
}
