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

#include "baggedLearner.h"
#include "RFDTree.h"
#include "learnerRegistry.h"
#include "StoredInstanceStream.h"
#include "StoredIndirectInstanceStream.h"
#include "utils.h"
#include "mtrand.h"

#include <assert.h>
#include <float.h>
#include <stdlib.h>



BaggedLearner::BaggedLearner(char*const*& argv, char*const* end) : learnerName_(NULL)
{ learner *theLearner = NULL;

  name_ = "BAGGED";

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


BaggedLearner::~BaggedLearner(void) {
  for (unsigned int i = 0; i < classifiers_.size(); ++i) {
    delete classifiers_[i];
  }
}

void  BaggedLearner::getCapabilities(capabilities &c){
  c = capabilities_;
}

void BaggedLearner::train(InstanceStream &is) {



	// load the data into a store
  StoredInstanceStream store;
  StoredIndirectInstanceStream thisStream;  ///< the bagged stream for learning the next classifier
  AddressableInstanceStream* aStream = dynamic_cast<AddressableInstanceStream*>(&is);
  MTRand_int32 rand;                       ///< random number generator for selecting bags

//  double wei[2];
//  wei[0]=0.2;
//  wei[1]=0.8;
//
//  MTRand ran();
//
//for(unsigned int i=0;i<10;i++)
//	printf("%f\n",ran());
//


  if (aStream == NULL) {
    store.setSource(is);
    aStream = &store;
  }

  // reset the classifier
  for (unsigned int i = 0; i < classifiers_.size(); ++i) delete classifiers_[i];
  classifiers_.clear();

  const InstanceCount dataSize = aStream->size();

  for (unsigned int i = 0; i < size_; ++i) {
    thisStream.setSourceWithoutLoading(*aStream);  // clear the stream

    for (InstanceCount instCount = 0; instCount < dataSize; ++instCount) {
      aStream->goTo(rand(dataSize)+1);
      thisStream.add(aStream->current());
    }

    classifiers_.push_back(createLearner(learnerName_, learnerArgv_, learnerArgEnd_));

    classifiers_.back()->train(thisStream);
  }
}


void BaggedLearner::classify(const instance &inst, std::vector<double> &classDist) {
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



