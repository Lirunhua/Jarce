/* Open source system for classification learning from very large data
** Class for creating an Attribute x Attribute crosstabulation of conditional (on class) mutual information from an xxyDist
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
#include "crosstab.h"
#include "xxyDist.h"

class cmiCrossTab : public crosstab<float> {
public:
  cmiCrossTab(xxyDist &dist, const bool discrVals = false);
  ~cmiCrossTab(void);
};
