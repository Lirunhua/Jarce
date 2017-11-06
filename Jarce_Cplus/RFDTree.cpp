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
#include "RFDTree.h"
#include "StoredInstanceStream.h"
#include "StoredIndirectInstanceStream.h"
#include "correlationMeasures.h"
#include "utils.h"
#include <math.h>
#include <limits>
#include <algorithm>

char*const* nullPtr = NULL;

RFDTree::RFDTree(char*const*& argv, char*const* end)
: trainingParams_(10,1.0,10), DTree(nullPtr, nullPtr)
{ name_ = "RFDTree";

  // get arguments
  while (argv != end) {
    if (*argv[0] != '-') {
      break;
    }
    else if (argv[0][1] == 's') {
      getUIntFromStr(argv[0]+2, trainingParams_.minLeafSize_, "minimum leaf size");
    }
    else if (argv[0][1] == 'm') {
      getUIntFromStr(argv[0]+2, classificationParams_.m_, "m");
      trainingParams_.m_ = classificationParams_.m_;
    }
    else if (argv[0][1] == 'n') {
      getUIntFromStr(argv[0]+2, trainingParams_.noOfAttsConsidered_, "n");
    }
    else {
      break;
    }

    name_ += argv[0];

    ++argv;
  }
}

RFDTree::~RFDTree(void)
{
}

static const NumValue UNKNOWN = std::numeric_limits<NumValue>::max();  ///< indicator for an unknown numeric value

class candidate {
public:
  candidate(AttType t, unsigned int i);
  AttType type;
  unsigned int index;
};

candidate::candidate(AttType t, unsigned int i) : type(t), index(i) {}


/// grow a sub tree from the instances
DTNode* RFDTree::growTree(IndirectInstanceStream &instances,
                        DTTrainingParams *params,
                        std::set<CategoricalAttribute> &blockedCatAtts,
                        std::set<NumericAttribute> &blockedNumAtts,
                        const std::vector<float> &priors
                        ) {
  std::vector<float> thisPriors;
  const double Epsilon = std::numeric_limits<double>::epsilon();
  const double minGain = Epsilon; // in C4.5 this is set to the average gain for all attributes
  const unsigned int noOfClasses = instances.getNoClasses();
  instances.rewind();

  const InstanceCount noOfInstances = instances.size();

  // get the joint distribution from the data
  xyDist dist(&instances);

  // check whether the data all belong to one class
  bool isPure = false;

  for (CatValue y = 0; y < noOfClasses; ++y) {
    const InstanceCount count = dist.getClassCount(y);

    thisPriors.push_back(static_cast<float>(count + priors[y] * params->m_));

    if (count == noOfInstances) {
      isPure = true;
    }
  }

  normalise(thisPriors);

  if (noOfInstances < 2 * params->minLeafSize_) {
    return new DTLeaf(thisPriors, noOfInstances);
  }

  if (isPure) {
    return new DTLeaf(thisPriors, noOfInstances);
  }

  // otherwise, find the attribute with the best info gain
  double bestGR = 0.0;
  CategoricalAttribute bestCatAtt = 0;
  NumericAttribute bestNumAtt = 0;
  NumValue bestCut;
  bool bestAttSet = false;
  bool bestAttIsNumeric = false;
  bool allAttsHaveManyValues = true;  ///< true iff all categorical attributes have too many values to be selected in normal processing.
  std::vector<candidate> candidates;

  for (CategoricalAttribute a = 0; a < instances.getNoCatAtts(); ++a) {
    candidate c(CATEGORICAL, a);
    if (blockedNumAtts.find(a) == blockedNumAtts.end()) candidates.push_back(c);
  }

  for (NumericAttribute a = 0; a < instances.getNoNumAtts(); ++a) {
    candidate c(NUMERIC, a);
    if (blockedNumAtts.find(a) == blockedNumAtts.end()) candidates.push_back(c);
  }

  std::random_shuffle(candidates.begin(), candidates.end());

  RFDTTrainingParams* rfdtParams = dynamic_cast<RFDTTrainingParams*>(params);

  if (rfdtParams == NULL) error("Internal Error: Error recasting RFDT params");

  const unsigned int noOfCandidates = min(rfdtParams->noOfAttsConsidered_, static_cast<unsigned int>(candidates.size()));

  for (unsigned int i = 0; i < noOfCandidates; ++i) {
    if (candidates[i].type == CATEGORICAL) {
      if (static_cast<double>(instances.getNoValues(candidates[i].index)) < 0.3 * noOfInstances) {
        allAttsHaveManyValues = false;
        break;
      }
    }
  }

  for (unsigned int i = 0; i < noOfCandidates; ++i) {
    if (candidates[i].type == CATEGORICAL) {
      CategoricalAttribute a = candidates[i].index;

      // check that the attribute does not have too many values and is not blocked and if not check if it has the best gain
      if ((allAttsHaveManyValues || static_cast<double>(instances.getNoValues(a)) < 0.3 * noOfInstances) &&
          blockedCatAtts.find(a) == blockedCatAtts.end() &&
          atLeastTwoValsAreSufficientlyFrequent(dist, a, params->minLeafSize_)) {
        const double gain = getInfoGain(dist, a);
        
        if (gain > minGain) {
          const double info = getInformation(dist, a);

          if (info > Epsilon) {
            const double gr = gain / info;
            if (gr > bestGR) {
              bestGR = gr;
              bestCatAtt = a;
              bestAttSet = true;
              bestAttIsNumeric = false;
            }
          }
        }
      }
    }
    else {
      assert(candidates[i].type == NUMERIC);
      NumericAttribute a = candidates[i].index;
      StoredIndirectInstanceStream knownInstances;                // the instances for which the value is known
      StoredIndirectInstanceStream unknownInstances;              // the instances for which the value is not known
      std::vector<InstanceCount> leftCounts;                // the class distribution for a left branch
      std::vector<InstanceCount> rightCounts;               // the class distribution for a right branch
      std::vector<InstanceCount> unknownCounts;             // the class distribution for the instances with missing values
      std::vector<InstanceCount> thisCounts;                // the counts for the current value

      if (blockedNumAtts.find(a) == blockedNumAtts.end()) {
        rightCounts.assign(noOfClasses, 0);

        // separate the known and unknown instances
        knownInstances.setSourceWithoutLoading(instances);    // clear the stream
        unknownInstances.setSourceWithoutLoading(instances);  // clear the stream
        instances.rewind();

        while (instances.advance()) {
          instance* inst = instances.current();

          if (inst->isMissing(a)) unknownInstances.add(inst);
          else {
            knownInstances.add(inst);                                 // add the instance to the set of known instances
            const CatValue y = inst->getClass();  // initially all instances are on the right branch, so initialise the right distribution
            ++rightCounts[y];
          }
        }

        thisCounts.assign(noOfClasses, 0);

        if (knownInstances.size() > 2 * params->minLeafSize_) {
          // do nothing if all instances are unknown
          // get the class distribution for the unknown instances
          unknownInstances.rewind();
          unknownCounts.assign(noOfClasses, 0);

          while (unknownInstances.advance()) {
            const CatValue y = unknownInstances.current()->getClass();
            ++unknownCounts[y];
          }

          // now check each possible split on the known instances
          knownInstances.sort(a);

          leftCounts.assign(noOfClasses, 0);

          // do not consider cuts until the size is at least minLeafSize_
          InstanceCount count = 0;
          InstanceCount lastCount = 0;

          NumValue thisVal = UNKNOWN;   // the current value
          NumValue lastVal = UNKNOWN;   // the last value
          CatValue lastY = noOfClasses; // the most common class for the last value - initially set to a non-class

          while (knownInstances.advance()) {
            const CatValue y = knownInstances.current()->getClass();
            const NumValue v = knownInstances.current()->getNumVal(a);

            ++count;

            if (v != thisVal) {
              // try the cut for thisVal
              const CatValue thisY = indexOfMaxVal(thisCounts); // the most common class for the value

              if (thisY != lastY && lastCount > params->minLeafSize_) {
                double gr = getGainRatio(leftCounts, rightCounts, unknownCounts);

                if (gr > bestGR) {
                  bestGR = gr;
                  bestNumAtt = a;
                  bestAttSet = true;
                  bestAttIsNumeric = true;
                  bestCut = lastVal + (thisVal-lastVal)/2.0;  // put the cut half way between values
                }
              }

              if (count == 1) {
                lastVal = UNKNOWN;   // no last value
                lastY = noOfClasses; // no last class
              }
              else {
                lastY = thisY;
                lastVal = thisVal;
                lastCount = count;

                for (CatValue y = 0; y < noOfClasses; ++y) {
                  leftCounts[y] += thisCounts[y];
                  rightCounts[y] -= thisCounts[y];
                  thisCounts[y] = 0;
                }
              }

              thisVal = v;
            }

            ++thisCounts[y];
          }
        }
      }
    }
  }

  // make the best split and grow the subtrees
  if (bestAttSet) {
    if (bestAttIsNumeric) {
      return new DTNumNode(bestNumAtt, bestCut, instances, params, blockedCatAtts, blockedNumAtts, thisPriors, growTree);
    }
    else { // the best attribute is categorical
      return new DTCatNode(bestCatAtt, instances, params, blockedCatAtts, blockedNumAtts, thisPriors, growTree);
    }
  }
  else {
    // no attribute on which to split
    return new DTLeaf(thisPriors, noOfInstances);
  }
}


//////////////////////
// The learner methods
//////////////////////

/// train the classifier from an instance stream
void RFDTree::train(InstanceStream &is) {
  
  testCapabilities(is);
  
  source_ = &is;

  const unsigned int noOfClasses = is.getNoClasses();
  
  std::vector<float> priors; ///< the prior probability of each class

  for (CatValue y = 0; y < noOfClasses; ++y) {
    priors.push_back(1.0/noOfClasses);
  }

  StoredInstanceStream store;
  StoredIndirectInstanceStream indirectStream;

  // check that the stream is an indirect instance stream.
  // we want an indirect stream to make sorting efficient.
  IndirectInstanceStream* theStream = dynamic_cast<IndirectInstanceStream*>(&is);

  if (theStream == NULL) {
    // if not, first check whether it is an addressable instance stream
    AddressableInstanceStream* adStream = dynamic_cast<AddressableInstanceStream*>(&is);

    if (theStream == NULL) {
      // if not convert it to addressable
      store.setSource(is);
      adStream = &store;
    }

    // now convert it to indirect
    indirectStream.setSource(*adStream);

    theStream = &indirectStream;
  }

  std::set<CategoricalAttribute> blockedCatAtts;
  std::set<NumericAttribute> blockedNumAtts;
  
  if (root_ != NULL) delete root_;
  
  root_ = growTree(*theStream, &trainingParams_, blockedCatAtts, blockedNumAtts, priors);  trainingIsFinished_ = true;
}

/// infer the class distribution for the current instance in the instance stream
void RFDTree::classify(const instance &inst, std::vector<double> &classDist) {
  root_->classify(inst, classDist);
}

/// describes what kind of data the learner is able to handle
void RFDTree::getCapabilities(capabilities &c) {
  c.setCatAtts(true);
  c.setNumAtts(true);
}

void RFDTree::printClassifier() {
  root_->print(0, *source_);
  putchar('\n');
}

