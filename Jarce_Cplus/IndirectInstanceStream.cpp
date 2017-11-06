/* Open source system for classification learning from very large data
** Class for an input stream that is a sequence of pointers to instances.
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
#include "IndirectInstanceStream.h"
#include "IndirectInstanceSubstream.h"
#include <algorithm>
#include <assert.h>

/// create the instance stream without any initialisation.  Must call setSource or setSourceWithoutLoading before use
IndirectInstanceStream::IndirectInstanceStream()
{
}

IndirectInstanceStream::~IndirectInstanceStream(void)
{
}

/// randomise the order in which instances will be accessed. Note, this does not affect any instances that have already been accessed.
void IndirectInstanceStream::shuffle() {
  std::random_shuffle(start_, end_);
}


// creates a comparator for two attributes based on their relative mutual information with the class
class CmpClass {
public:
  CmpClass(const NumericAttribute att) : att_(att) {
     }

  bool operator() (instance *a, instance *b) {
      return a->getNumVal(att_) < b->getNumVal(att_);
    }

  private:
    const NumericAttribute att_;
};

void IndirectInstanceStream::sort(const NumericAttribute att) {

  CmpClass cmp(att);

  std::sort(start_, end_, cmp);

  rewind();
}

void IndirectInstanceStream::rewind() {
  current_ = start_-1;
}

InstanceCount IndirectInstanceStream::size() {
  return end_ - start_;
}

/// advance, discarding the next instance in the stream.  Return true iff successful.
bool IndirectInstanceStream::advance() {
  ++current_;
  return current_ < end_;
}

/// set the current instance to the one at the specified position in the stream
/// indexes start at 1
void IndirectInstanceStream::goTo(InstanceCount position) {
  current_ = start_ + position - 1;
}

/// get a pointer to the current instance.
/// Requires that there be a current instance, so must either check isAtEnd or the the most recent advance was successful.
instance* IndirectInstanceStream::current() {
  assert(current_ >= start_ && current_ < end_);
  return *current_;
}

/// advance to the next instance in the stream. Return true iff successful. @param inst the instance record to receive the new instance. 
bool IndirectInstanceStream::advance(instance &inst) {
  ++current_;
  if (current_ >= end_) return false;
  else {
    inst = **current_;
    return true;
  }
}

bool IndirectInstanceStream::isAtEnd() {
  return current_ >= end_;
}


void IndirectInstanceStream::setIndirectInstanceSubstream(IndirectInstanceSubstream &substream, InstanceCount start, InstanceCount size) {
  substream.setSubstream(*source_, start_+start, std::min(size, static_cast<InstanceCount>(end_-start_-start)));
}
