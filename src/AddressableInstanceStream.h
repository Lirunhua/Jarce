/* Open source system for classification learning from very large data
** Abstract class for an instance stream which provides a pointer to the current instance
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

#include "instanceStreamFilter.h"

class AddressableInstanceStream: public InstanceStreamFilter
{
public:
  AddressableInstanceStream() {}
  ~AddressableInstanceStream(void) {}

  virtual instance* current() = 0;                ///< get a pointer to the current instance

  virtual void goTo(InstanceCount position) = 0;  ///< set the current instance to the one at the specified position in the stream, indexes start at 1
};
