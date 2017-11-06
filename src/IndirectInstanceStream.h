/* Open source system for classification learning from very large data
** Class for an input stream of randomly sampled instances.  Instances are stored as pointers rather than copies.
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
#include "AddressableInstanceStream.h"
#include "instanceStreamFilter.h"
#include "mtrand.h"

#include <vector>

class IndirectInstanceSubstream;

class IndirectInstanceStream : public AddressableInstanceStream
{
public:
  IndirectInstanceStream();
  ~IndirectInstanceStream(void);

  void rewind();                                              ///< return to the first instance in the stream
  bool advance();                                             ///< advance, discarding the next instance in the stream.  Return true iff successful.
  bool advance(instance &inst);                               ///< advance to the next instance in the stream.  Return true iff successful. @param inst the instance record to receive the new instance. 
  bool isAtEnd();                                             ///< true if we have advanced past the last instance
  InstanceCount size();                                       ///< the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.  The stream state is undefined after a call to size(), so a rewind shouldbe performed before the next advance.

  instance* current() ;                                       ///< get a pointer to the current instance. Requires that there be a current instance, so must either check isAtEnd or the the most recent advance was successful.
  void goTo(InstanceCount position);                          ///< move to the specified position in the stream

  void shuffle();                                             ///< randomise the order in which instances will be accessed. Note, this does not affect any instances that have already been accessed.
  void sort(const NumericAttribute att);                      ///< sort the instances into ascending order on the attribute

  void setIndirectInstanceSubstream(IndirectInstanceSubstream &substream, InstanceCount start, InstanceCount size);

protected:
  instance** start_;      ///< the first instance pointer
  instance** current_;    ///< the current instance pointer
  instance** end_;        ///< one past the last instance pointer
};
