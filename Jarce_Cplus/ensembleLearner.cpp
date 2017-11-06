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

#include "ensembleLearner.h"
#include "RFDTree.h"
#include "learnerRegistry.h"
#include "StoredInstanceStream.h"
#include "StoredIndirectInstanceStream.h"
#include "utils.h"
#include "mtrand.h"

#include <assert.h>
#include <float.h>
#include <stdlib.h>



EnsembleLearner::EnsembleLearner(char*const*& argv, char*const* end) : learnerName_(NULL)
{ learner *theLearner = NULL;

  name_ = "Ensembled";

  // defaults
  useMajorityVoting_ = true;
  size_ = 100;
  
  // get arguments
  while (argv != end) {
    if (*argv[0] != '-') {
      break;
    }
    else if (argv[0][1] == 'd') {
      useMajorityVoting_ = false;;

      name_ += argv[0];
    }
    else if (argv[0][1] == 'b') {
      // specify the base learner
      learnerName_ = argv[0]+2;
      learnerArgv_ = ++argv;
  
      // create the learner
      theLearner = createLearner(learnerName_, argv, end);
      
      if (theLearner == NULL) {
        error("Learner %s is not supported", learnerName_);
      }

      learnerArgEnd_ = argv;

      name_ += "_";
      name_ += *theLearner->getName();
      break;
    }
    else if (argv[0][1] == 's') {
      getUIntFromStr(argv[0]+2, size_, "s");

      name_ += argv[0];
    }
    else {
      break;
    }

    ++argv;
  }

  if (theLearner == NULL) error("No base learner specified");
  else {
    theLearner->getCapabilities(capabilities_);
    delete theLearner;
  }
}


EnsembleLearner::~EnsembleLearner(void) {
  for (unsigned int i = 0; i < classifiers_.size(); ++i) {
    delete classifiers_[i];
  }
}

void  EnsembleLearner::getCapabilities(capabilities &c){
  c = capabilities_;
}

void EnsembleLearner::train(InstanceStream &is) {
  // reset the classifier
  for (unsigned int i = 0; i < classifiers_.size(); ++i) delete classifiers_[i];
  classifiers_.clear();

  // train a new classifier
  for (unsigned int i = 0; i < size_; ++i) {
    classifiers_.push_back(createLearner(learnerName_, learnerArgv_, learnerArgEnd_));
    classifiers_.back()->train(is);
  }
}


void EnsembleLearner::classify(const instance &inst, std::vector<double> &classDist) {
  std::vector<double> thisClassDist(classDist.size());
  
  classDist.assign(classDist.size(), 0.0);

  for (unsigned int i = 0; i < classifiers_.size(); ++i) {
    classifiers_[i]->classify(inst, thisClassDist);

    if (useMajorityVoting_) {
      //double maxVal = max(thisClassDist);
      classDist[indexOfMaxVal(thisClassDist)] += 1.0;
    }
    else {
      for (CatValue y = 0; y < classDist.size(); ++y) {
        classDist[y] += thisClassDist[y];
      }
    }
  }

  normalise(classDist);
}



