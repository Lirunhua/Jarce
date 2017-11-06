/* Open source system for classification learning from very large data
** Class for an input stream in which instances are stored in core
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

class StoredInstanceStream : public AddressableInstanceStream
{
public:
  StoredInstanceStream();
  ~StoredInstanceStream(void);

  void rewind();                                              ///< return to the first instance in the stream
  bool advance();                                             ///< advance, discarding the next instance in the stream.  Return true iff successful.
  bool advance(instance &inst);                               ///< advance to the next instance in the stream.  Return true iff successful. @param inst the instance record to receive the new instance. 
  bool isAtEnd();                                             ///< true if we have advanced past the last instance
  InstanceCount size();                                       /// the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.  The stream state is undefined after a call to size(), so a rewind shouldbe performed before the next advance.

  unsigned int getNoClasses() const;                          ///< return the number of classes
  const char* getClassName(CatValue att) const;               ///< return the name for a class
  const char* getClassAttName() const;                        ///< return the name for the class attribute
  unsigned int getNoCatAtts() const;                          ///< return the number of categorical attributes
  unsigned int getNoValues(CategoricalAttribute att) const;   ///< return the number of values for a categorical attribute
  const char* getCatAttName(CategoricalAttribute att) const;  ///< return the name for a categorical Attribute
  const char* getCatAttValName(CategoricalAttribute att, CatValue val) const; ///< return the name for a categorical attribute value
  unsigned int getNoNumAtts() const;                          ///< return the number of numeric attributes
  const char* getNumAttName(NumericAttribute att) const;      ///< return the name for a numeric attribute
  unsigned int getPrecision(NumericAttribute att) const;      ///< return the precision to which values of a numeric attribute should be output
  const char* getName();                                      ///< return a string that gives a meaningful name for the stream
  bool areNamesCaseSensitive();                               ///< true iff name comparisons are case sensitive

  void setSource(InstanceStream &source);                     ///< set the source for the sample

  instance* current();                                        ///< get a pointer to the current instance. Requires that there be a current instance, so must either check isAtEnd or the the most recent advance was successful.
  void goTo(InstanceCount position);                          ///< move to the specified position in the stream

private:
  std::vector<instance> store_; ///< the stored sample
  unsigned int next_;                ///< index of the next instance to be retrieved by advance. Use index rather than iterator to avoid problems with the iterator becoming invalid
};
