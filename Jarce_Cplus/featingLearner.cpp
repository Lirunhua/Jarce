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

#include "featingLearner.h"
#include "DTree.h"
#include "learnerRegistry.h"
#include "StoredInstanceStream.h"
#include "StoredIndirectInstanceStream.h"
#include "utils.h"
#include "mtrand.h"

#include <assert.h>
#include <float.h>
#include <stdlib.h>

static char*const* argBegin = NULL;
static char*const* argEnd = argBegin;

FeatingLearner::FeatingLearner(char*const*& argv, char*const* end) : learnerName_(NULL), discretiser_("mdl", argBegin, argEnd)
{ learner *theLearner = NULL;

  name_ = "FEATED";

  // defaults
  useMajorityVoting_ = true;
  
  // get arguments
  while (argv != end) {
    if (*argv[0] != '-') {
      break;
    }
    else if (argv[0][1] == 'd') {
      useMajorityVoting_ = false;;
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
    else {
      break;
    }

    name_ += argv[0];

    ++argv;
  }

  if (theLearner == NULL) error("No base learner specified");
  else {
    theLearner->getCapabilities(capabilities_);
    delete theLearner;
  }
}


FeatingLearner::~FeatingLearner(void) {
  for (CategoricalAttribute a = 0; a < catClassifiers_.size(); ++a) {
    for (CatValue v = 0; v < catClassifiers_[a].size(); ++v) {
      delete catClassifiers_[a][v];
    }
  }
  for (NumericAttribute a = 0; a < numClassifiers_.size(); ++a) {
    for (CatValue v = 0; v < numClassifiers_[a].size(); ++v) {
      delete numClassifiers_[a][v];
    }
  }
}

void  FeatingLearner::getCapabilities(capabilities &c){
  c = capabilities_;
}

void FeatingLearner::train(InstanceStream &is) {
  // load the data into a store
  StoredInstanceStream store;
  StoredIndirectInstanceStream thisStream;  ///< the bagged stream for learning the next classifier
  AddressableInstanceStream* aStream = dynamic_cast<AddressableInstanceStream*>(&is);
  MTRand_int32 rand;                       ///< random number generator for selecting bags

  if (aStream == NULL) {
    store.setSource(is);
    aStream = &store;
  }

  for (CategoricalAttribute a = 0; a < catClassifiers_.size(); ++a) {
    for (CatValue v = 0; v < catClassifiers_[a].size(); ++v) {
      delete catClassifiers_[a][v];
    }
  }
  catClassifiers_.clear();
  catClassifiers_.resize(is.getNoCatAtts());

  for (NumericAttribute a = 0; a < numClassifiers_.size(); ++a) {
    for (CatValue v = 0; v < numClassifiers_[a].size(); ++v) {
      delete numClassifiers_[a][v];
    }
  }
  numClassifiers_.clear();
  numClassifiers_.resize(is.getNoNumAtts());

  for (CategoricalAttribute a = 0; a < aStream->getNoCatAtts(); ++a) {
    for (CatValue v = 0; v < aStream->getNoValues(a); ++v) {
      thisStream.setSourceWithoutLoading(*aStream);  // clear the stream

      aStream->rewind();

      while (aStream->advance()) {
        if (aStream->current()->getCatVal(a) == v) {
          thisStream.add(aStream->current());
        }
      }

      catClassifiers_[a].push_back(createLearner(learnerName_, learnerArgv_, learnerArgEnd_));
      catClassifiers_[a].back()->train(thisStream);
    }
  }

  if (aStream->getNoNumAtts() > 0) {
    discretiser_.setSource(*aStream);   // discretise the numeric attributes

    for (NumericAttribute a = 0; a < aStream->getNoNumAtts(); ++a) {
      CatValue v = 0;
      CatValue maxVal = 0;
      
      while (v <= maxVal) {
        thisStream.setSourceWithoutLoading(*aStream);  // clear the stream

        aStream->rewind();

        while (aStream->advance()) {
          CatValue thisVal = discretiser_.discretise(aStream->current()->getNumVal(a), a);

          if (thisVal == v) {
            thisStream.add(aStream->current());
          }
          else if (thisVal > maxVal) {
            maxVal = thisVal;
          }
        }

        numClassifiers_[a].push_back(createLearner(learnerName_, learnerArgv_, learnerArgEnd_));
        numClassifiers_[a].back()->train(thisStream);
        ++v;
      }
    }
  }
}


void FeatingLearner::classify(const instance &inst, std::vector<double> &classDist) {
  std::vector<double> thisClassDist(classDist.size());
  
  classDist.assign(classDist.size(), 0.0);

  for (CategoricalAttribute a = 0; a < catClassifiers_.size(); ++a) {
    catClassifiers_[a][inst.getCatVal(a)]->classify(inst, thisClassDist);

    if (useMajorityVoting_) {
      classDist[indexOfMaxVal(thisClassDist)] += 1.0;
    }
    else {
      for (CatValue y = 0; y < classDist.size(); ++y) {
        classDist[y] += thisClassDist[y];
      }
    }
  }

  for (NumericAttribute a = 0; a < numClassifiers_.size(); ++a) {
    const CatValue v = discretiser_.discretise(inst.getNumVal(a), a);

    if (v < numClassifiers_[a].size()) {  // check that the classifier exists.  It might not if a value was not encountered at training time.
      numClassifiers_[a][v]->classify(inst, thisClassDist);

      if (useMajorityVoting_) {
        classDist[indexOfMaxVal(thisClassDist)] += 1.0;
      }
      else {
        for (CatValue y = 0; y < classDist.size(); ++y) {
          classDist[y] += thisClassDist[y];
        }
      }
    }
  }

  normalise(classDist);
}



