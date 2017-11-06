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
#include "StoredInstanceStream.h"
#include <assert.h>

StoredInstanceStream::StoredInstanceStream() : next_(0)
{
}

StoredInstanceStream::~StoredInstanceStream(void)
{
}

void StoredInstanceStream::setSource(InstanceStream &source) {
  InstanceCount count = 1;
  
  source_ = &source;
  metaData_ = source.getMetaData();

  store_.resize(1);
  store_.back().init(source);
  
  // load the sample
  source.rewind();

  while (source.advance(store_.back())) {
    store_.resize(++count);
    store_.back().init(source);
  }

  store_.resize(--count);

  rewind();
}

void StoredInstanceStream::rewind() {
  next_ = 0;
}

InstanceCount StoredInstanceStream::size() {
  return store_.size();
}

/// advance, discarding the next instance in the stream.  Return true iff successful.
bool StoredInstanceStream::advance() {
  ++next_;
  return next_ <= store_.size();
}

/// get a pointer to the current instance.
/// Requires that there be a current instance, so must either check isAtEnd or the the most recent advance was successful.
instance* StoredInstanceStream::current() {
  assert(next_ > 0 && next_ <= store_.size());
  return &store_[next_-1];
}

/// advance to the next instance in the stream. Return true iff successful. @param inst the instance record to receive the new instance. 
bool StoredInstanceStream::advance(instance &inst) {
  if (next_ >= store_.size()) return false;
  else {
    inst = store_[next_];
    ++next_;
    return true;
  }
}


/// advance to the specified position in the stream.
void StoredInstanceStream::goTo(InstanceCount position) {
  next_ = position; // indexes start at 1, whereas next_ is indexed starting from 0
}


bool StoredInstanceStream::isAtEnd() {
  return next_ >= store_.size();
}

/// return the number of classes
unsigned int StoredInstanceStream::getNoClasses() const {
  return source_->getNoClasses();
}

/// return the name for a class
const char* StoredInstanceStream::getClassName(CatValue y) const {
  return source_->getClassName(y);
}

///< return the name for the class attribute
const char* StoredInstanceStream::getClassAttName() const {
  return source_->getClassAttName();
}

/// return the number of categorical attributes
unsigned int StoredInstanceStream::getNoCatAtts() const {
  return source_->getNoCatAtts();
}

/// return the number of values for a categorical attribute
unsigned int StoredInstanceStream::getNoValues(CategoricalAttribute att) const {
  return source_->getNoValues(att);
}

/// return the name for a categorical Attribute
const char* StoredInstanceStream::getCatAttName(CategoricalAttribute att) const {
  return source_->getCatAttName(att);
}

/// return the name for a categorical attribute value
const char* StoredInstanceStream::getCatAttValName(CategoricalAttribute att, CatValue val) const {
  return source_->getCatAttValName(att, val);
}

/// return the number of numeric attributes
unsigned int StoredInstanceStream::getNoNumAtts() const {
  return source_->getNoNumAtts();
}

/// return the name for a numeric attribute
const char* StoredInstanceStream::getNumAttName(CategoricalAttribute att) const {
  return source_->getNumAttName(att);
}

/// return the name for a numeric attribute
unsigned int StoredInstanceStream::getPrecision(NumericAttribute att) const {
  return source_->getPrecision(att);
}

/// return a string that gives a meaningful name for the stream
const char* StoredInstanceStream::getName() {
  return source_->getName();
}

///< true iff name comparisons are case sensitive
bool StoredInstanceStream::areNamesCaseSensitive() {
  return source_->areNamesCaseSensitive();
}
