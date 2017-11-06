/* Open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
** Implements Sahami's k-dependence Bayesian classifier
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
** Please report any bugs to Ana M. Martinez <anam.martinez@monash.edu>
*/
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>
#ifdef __linux__
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "kdbCondDisc2.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"
#include "instanceStreamDiscretiser.h"
#include "MDLDiscretiser.h"

kdbCondDisc2::kdbCondDisc2(char*const*& argv, char*const* end){
  name_ = "KDB-CondDisc2";

  // defaults
  k_ = 1;
  minCount_ = 0;
  
  // get arguments
  while (argv != end) {
    if (*argv[0] != '-') {
      break;
    }
    else if (argv[0][1] == 'k') {
      getUIntFromStr(argv[0]+2, k_, "k");
    }
    else if (argv[0][1] == 'm') {
     getUIntFromStr(argv[0]+2, minCount_, "m");
    }
    else {
      break;
    }

    name_ += argv[0];

    ++argv;
  }
}

kdbCondDisc2::~kdbCondDisc2(void)
{
}


void  kdbCondDisc2::getCapabilities(capabilities &c){
  c.setCatAtts(true);  
  c.setNumAtts(true);
}

// creates a comparator for two attributes based on their relative mutual information with the class
class miCmpClass {
public:
  miCmpClass(std::vector<float> *m) {
    mi = m;
  }

  bool operator() (CategoricalAttribute a, CategoricalAttribute b) {
    return (*mi)[a] > (*mi)[b];
  }

private:
  std::vector<float> *mi;
};

void kdbCondDisc2::reset(InstanceStream &is) {
  kdb::reset(is);
  noOrigCatAtts_ = static_cast<InstanceStreamDiscretiser::MetaData*>(static_cast<InstanceStreamDiscretiser&>(is).getMetaData())->getNoOrigCatAtts();
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
void kdbCondDisc2::train(const instance &inst) {
  
  instance instDisc(*instanceStream_);
  static_cast<InstanceStreamDiscretiser*>(instanceStream_)->discretiseInstance(inst, instDisc);
  
  if (pass_ == 1) {
    dist_.update(instDisc);
  }
  else {
    assert(pass_ == 2);

    for (CategoricalAttribute ca = 0; ca < noOrigCatAtts_; ca++) { //update counts as usual for the originally discrete attributes
      dTree_[ca].update(instDisc, ca, parents_[ca]);
    }
    for (CategoricalAttribute na = noOrigCatAtts_; na < noCatAtts_; na++) {
      dTree_[na].update(instDisc, na, parents_[na], inst.getNumVal(na-noOrigCatAtts_)); //update counts as usual when not all parents are considered (for smoothing purposes) and stores the numeric values on the leaves (conditioned on all parents)
    }
    classDist_.update(inst);
  }
}

void kdbCondDisc2::train(InstanceStream &is){
  instance inst;
  
  testCapabilities(is);
  
  reset(is);

  while (!trainingIsFinished()) {
    initialisePass();
    is.rewind();
    while (static_cast<InstanceStreamDiscretiser&>(is).advanceNumeric(inst)) {
      train(inst);
    }
    finalisePass();
  }
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
void kdbCondDisc2::initialisePass() {
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
void kdbCondDisc2::finalisePass() {
  if (pass_ == 1) {
    // calculate the mutual information from the xy distribution
    std::vector<float> mi;  
    getMutualInformation(dist_.xyCounts, mi);
    
    if (verbosity >= 3) {
      printf("\nMutual information table\n");
      print(mi);
    }

    // calculate the conditional mutual information from the xxy distribution
    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    getCondMutualInf(dist_,cmi);
    
    dist_.clear();

    if (verbosity >= 3) {
      printf("\nConditional mutual information table\n");
      cmi.print();
    }

    // sort the attributes on MI with the class
    std::vector<CategoricalAttribute> order;

    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
      order.push_back(a);
    }

    // assign the parents
    if (!order.empty()) {
      miCmpClass cmp(&mi);

      std::sort(order.begin(), order.end(), cmp);

      if (verbosity >= 2) {
        printf("\n%s parents:\n", instanceStream_->getCatAttName(order[0]));
      }

      // proper KDB assignment of parents
      for (std::vector<CategoricalAttribute>::const_iterator it = order.begin()+1; it != order.end(); it++) {
        parents_[*it].push_back(order[0]);
        for (std::vector<CategoricalAttribute>::const_iterator it2 = order.begin()+1; it2 != it; it2++) {
          // make parents into the top k attributes on mi that precede *it in order
          if (parents_[*it].size() < k_) {
            // create space for another parent
            // set it initially to the new parent.
            // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
            parents_[*it].push_back(*it2);
          }
          for (unsigned int i = 0; i < parents_[*it].size(); i++) {
            if (cmi[*it2][*it] > cmi[parents_[*it][i]][*it]) {
              // move lower value parents down in order
              for (unsigned int j = parents_[*it].size()-1; j > i; j--) {
                parents_[*it][j] = parents_[*it][j-1];
              }
              // insert the new att
              parents_[*it][i] = *it2;
              break;
            }
          }
        }

        if (verbosity >= 2) {
          printf("%s parents: ", instanceStream_->getCatAttName(*it));
          for (unsigned int i = 0; i < parents_[*it].size(); i++) {
            printf("%s ", instanceStream_->getCatAttName(parents_[*it][i]));
          }
          putchar('\n');
        }
      }
    }
  }
  else if (pass_ == 2) {
    
    // STEP 1: For all numeric attributes, go through all the parent-values combination and discretise the values.
    for (CategoricalAttribute na = noOrigCatAtts_; na < noCatAtts_; na++) {
      dtNode *dt = dTree_[na].getdTNode();
      CategoricalAttribute att = dt->att;
      if (att != NOPARENT) {//if the numeric attribute has no parents nothing is done
        // find all the leaves
        for (CatValue v = 0; v < dt->children.size(); v++) {
          std::vector<CatValue> parentValues;
          dtNode *next = dt->children[v];
          parentValues.push_back(v);
          leafSearch(next, na, parentValues);
        }
      }
    }
  }
  ++pass_;
}

void kdbCondDisc2::leafSearch(dtNode *next, const CategoricalAttribute na, std::vector<CatValue> parentValues){
  
    if (next == NULL)
      return;  
    dtNode *dt = next;
    CategoricalAttribute att = dt->att;
    
    if (att != NOPARENT) {
      // find all the leaves
      for (CatValue v = 0; v < dt->children.size(); v++) {
        dtNode *next = dt->children[v];
        parentValues.push_back(v);
        leafSearch(next, na, parentValues);
        parentValues.pop_back();
      }
    }else{
      //Here we get the numeric values
      if(dt->numValues_.size()!=0){//This test should be redundant
        std::vector<NumValue> vals;
        std::vector<CatValue> classes;
        std::vector<NumValue> cuts; 
        
        unsigned int noOfClasses = instanceStream_->getNoClasses();
        for (CatValue y = 0; y < noOfClasses; y++) {
          classes.insert(classes.end(),dt->numValues_[y].size(),y);
          vals.insert(vals.end(),dt->numValues_[y].begin(),dt->numValues_[y].end());
        }
        dt->numValues_.clear();
        
        // STEP 2: The data for a combination of parent-values is stored in vector vals, now find the appropriate cuts
        //        MDLDiscretiser *theDiscretiser = new MDLDiscretiser();
        //        theDiscretiser->discretise(vals, classes, instanceStream_->getNoClasses(), cuts);
        //        delete theDiscretiser;
        MDLDiscretiser *theDiscretiser = new MDLDiscretiser();
        theDiscretiser->discretise(vals, classes, noOfClasses, cuts);
        delete theDiscretiser;
        
        // STEP 3: Discretise values according to cuts
        std::vector<CatValue> valsDisc;
        discretise(vals, valsDisc, cuts); //This could be done more efficiently by ordering vals first.
        
        // STEP 4: Update the probabilities and save cuts on the appropriate leaves of the dTree.
        dTree_[na].update(valsDisc, na, classes, parents_[na], parentValues, cuts, noOrigCatAtts_);
      }
    }
}

void kdbCondDisc2::discretise(std::vector<NumValue> &vals, std::vector<CatValue> &valsDisc, std::vector<NumValue> &cuts) {
  for (std::vector<NumValue>::const_iterator it = vals.begin(); it != vals.end(); it++) {
      if (*it == MISSINGNUM) {
        valsDisc.push_back(cuts.size()+1);
      }
      else if (cuts.size() == 0) {
        valsDisc.push_back(0);
      }
      else if (*it > cuts.back()) {
        valsDisc.push_back(cuts.size());
      }
      else {
        unsigned int upper = cuts.size()-1;
        unsigned int lower = 0;

        while (upper > lower) {
          const unsigned int mid = lower + (upper-lower) / 2;

          if (*it <= cuts[mid]) {
            upper = mid;
          }
          else {
            lower = mid+1;
          }
        }

        assert(upper == lower);
        valsDisc.push_back(upper);
      }
  }
}


/// true iff no more passes are required. updated by finalisePass()
bool kdbCondDisc2::trainingIsFinished() {
  return pass_ > 2;
}

void kdbCondDisc2::classify(const instance& inst, std::vector<double> &posteriorDist) {
  // calculate the class probabilities in parallel
  // P(y)
  for (CatValue y = 0; y < noClasses_; y++) {
    posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
  }

  instance instDisc(*instanceStream_);
  static_cast<InstanceStreamDiscretiser*>(instanceStream_)->discretiseInstance(inst, instDisc);
  
  // P(x_i | x_p1, .. x_pk, y)
  for (CategoricalAttribute x = 0; x < noOrigCatAtts_; x++) {
      //Discretize inst appropriately according to multi conditional discretization for the numeric children and original discretization for the parents
    if (minCount_) {
      dTree_[x].updateClassDistribution(posteriorDist, x, instDisc, minCount_);
    }
    else{
      dTree_[x].updateClassDistribution(posteriorDist, x, instDisc);
    }
  }
  for (CategoricalAttribute x = noOrigCatAtts_; x < noCatAtts_; x++) {
    if(parents_[x].size()==0){
      if (minCount_) {
        dTree_[x].updateClassDistribution(posteriorDist, x, instDisc,minCount_);
      }
      else{
        dTree_[x].updateClassDistribution(posteriorDist, x, instDisc);
      }
    }else{
      if (minCount_) {
        dTree_[x].updateClassDistributionAndDiscAttValue(posteriorDist, x, instDisc, inst.getNumVal(x-noOrigCatAtts_), noOrigCatAtts_, minCount_);
      }else{
        dTree_[x].updateClassDistributionAndDiscAttValue(posteriorDist, x, instDisc, inst.getNumVal(x-noOrigCatAtts_), noOrigCatAtts_);
      }
    }
  }
  // normalise the results
  normalise(posteriorDist);
}




