/* Open source system for classification learning from very large data
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
#include "random.h"
#include <assert.h>
#include "utils.h"
#include <stdlib.h>

randomClassifier::randomClassifier(char*const*& argv, char*const* end)
{ 
  name_ = "Random Classifier";
}

randomClassifier::~randomClassifier(void)
{
}

void  randomClassifier::getCapabilities(capabilities &c){
  c.setCatAtts(true);  // only categorical attributes are supported at the moment
}

void randomClassifier::reset(InstanceStream&) {
}


void randomClassifier::train(const instance&) {
}


void randomClassifier::initialisePass() {
}


void randomClassifier::finalisePass() {
}


bool randomClassifier::trainingIsFinished() {
  return true;
}

void randomClassifier::train(InstanceStream&) {
}

void randomClassifier::classify(const instance&, std::vector<double> &classDist) {
  const unsigned int noClasses = classDist.size();

  for (CatValue y = 0; y < noClasses; y++) {
    classDist[y] = 0.0;
  }

  classDist[rand()%noClasses] = 1.0;
}
