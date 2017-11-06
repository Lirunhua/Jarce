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
#include "IndirectInstanceSample.h"
#include <algorithm>
#include <assert.h>

IndirectInstanceSample::IndirectInstanceSample(const unsigned int size) : targetSize_(size), poolSize_(0), next_(0)
{ store_.reserve(size);
}

IndirectInstanceSample::~IndirectInstanceSample(void)
{
}

void IndirectInstanceSample::setSource(InstanceStream &source) {
  source_ = &source;
  metaData_ = source.getMetaData();
  store_.clear();
  poolSize_ = 0;
  rewind();
}

/// randomise the order in which instances will be accessed. Note, this does not affect any instances that have already been accessed.
void IndirectInstanceSample::shuffle() {
  random_shuffle(store_.begin(), store_.end());
}

void IndirectInstanceSample::rewind() {
  next_ = 0;
}

// add the instance to the pool of vailable instances.
// instances are randomly sampled from the pool - do this as they are added
void IndirectInstanceSample::sample(instance &i) {
  ++poolSize_;
  if (store_.size() < targetSize_) store_.push_back(&i);
  else {
    unsigned int index = rand_(poolSize_);
    if (index < targetSize_) store_[index] = &i;
  }
}

InstanceCount IndirectInstanceSample::size() {
  return store_.size();
}

/// advance, discarding the next instance in the stream.  Return true iff successful.
bool IndirectInstanceSample::advance() {
  ++next_;
  return next_ <= store_.size();
}

/// get a pointer to the current instance.
/// Requires that there be a current instance, so must either check isAtEnd or the the most recent advance was successful.
instance* IndirectInstanceSample::current() {
  assert(next_ <= store_.size());
  return store_[next_-1];
}

/// advance to the next instance in the stream. Return true iff successful. @param inst the instance record to receive the new instance. 
bool IndirectInstanceSample::advance(instance &inst) {
  if (next_ > store_.size()) return false;
  else {
    inst = *store_[next_];
    ++next_;
    return true;
  }
}

bool IndirectInstanceSample::isAtEnd() {
  return next_ > store_.size();
}

/// return a string that gives a meaningful name for the stream
const char* IndirectInstanceSample::getName() {
  return "Sample";
}
