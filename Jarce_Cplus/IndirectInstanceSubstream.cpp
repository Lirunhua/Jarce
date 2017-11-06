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
#include "IndirectInstanceSubstream.h"
#include <algorithm>
#include <assert.h>

/// create the instance stream without any initialisation.  Must call setSource or setSourceWithoutLoading before use
IndirectInstanceSubstream::IndirectInstanceSubstream()
{
}

IndirectInstanceSubstream::~IndirectInstanceSubstream(void)
{
}

/// load a sample from an instance stream
void IndirectInstanceSubstream::setSubstream(InstanceStream &source, instance** start, InstanceCount size)
{ start_ = start;
  current_ = start;
  end_ = start_+size;
  setSource(source);
}
