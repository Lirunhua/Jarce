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
#include "tan_gen.h"
#include "utils.h"
#include "correlationMeasures.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>

TAN_gen::TAN_gen() :
		trainingIsFinished_(false) {
}

TAN_gen::TAN_gen(char* const *&, char* const *) :
		xxyDist_(), trainingIsFinished_(false) {
	name_ = "TAN";
}

TAN_gen::~TAN_gen(void) {}

void TAN_gen::reset(InstanceStream &is) {
	instanceStream_ = &is;
	const unsigned int noCatAtts = is.getNoCatAtts();
	noCatAtts_ = noCatAtts;
	noClasses_ = is.getNoClasses();

	trainingIsFinished_ = false;

	//safeAlloc(parents, noCatAtts_);
        parents_.resize(noCatAtts);
	for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
		parents_[a] = NOPARENT;
	}

	xxyDist_.reset(is);
}

void TAN_gen::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void TAN_gen::initialisePass() {
	assert(trainingIsFinished_ == false);
	//learner::initialisePass (pass_);
//	dist->clear();
//	for (CategoricalAttribute a = 0; a < meta->noCatAtts; a++) {
//		parents_[a] = NOPARENT;
//	}
}

void TAN_gen::train(const instance &inst) {
	xxyDist_.update(inst);
}

void TAN_gen::classify(const instance &inst, std::vector<double> &classDist) {
//
    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    const double totalCount = xxyDist_.xyCounts.count;
    
    std::vector<bool> generalizationSet;
    generalizationSet.assign(noCatAtts_, false);
    
    	for (CategoricalAttribute i = 1; i < noCatAtts_; i++) {
			const CatValue iVal = inst.getCatVal(i);
			const InstanceCount countOfxi = xxyDist_.xyCounts.getCount(i, iVal);

			for (CategoricalAttribute j = 0; j < i; j++) {
				if (!generalizationSet[j]) {
					const CatValue jVal = inst.getCatVal(j);
					const InstanceCount countOfxixj = xxyDist_.getCount(i, iVal,
							j, jVal);
					const InstanceCount countOfxj =xxyDist_.xyCounts.getCount(
							j, jVal);

					if (countOfxj == countOfxixj && countOfxj >= 30) {
						//xi is a generalisation or substitution of xj
						//once one xj has been found for xi, stop for rest j
						generalizationSet[i] = true;
						break;
					} else if (countOfxi == countOfxixj && countOfxi >= 30) {
						//xj is a generalisation of xi
						generalizationSet[j] = true;
					}
				}
			}
		}
    
    for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
      for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
        float m=0.0;
        // if (generalizationSet[x1]== false && generalizationSet[x2]=false) {
          unsigned int v1=inst.getCatVal(x1);
          unsigned int v2=inst.getCatVal(x2);
           for (CatValue y = 0; y < noClasses_; y++) {
            //  const double jx1x2y = xxyDist_.jointP(x1,v1,x2,v2,y);
              const double x1y = xxyDist_.xyCounts.p(x1,v1,y);
              const double x1x2y = xxyDist_.jointP(x1,v1,x2,v2,y)/xxyDist_.xyCounts.jointP(x2,v2,y);
              const double x2y = xxyDist_.xyCounts.p(x2,v2,y);
              const double x2x1y = xxyDist_.jointP(x1,v1,x2,v2,y)/xxyDist_.xyCounts.jointP(x1,v1,y);
              if (x1x2y) {
                 m+= x1y*log2(x1y)-x1x2y*log2(x1x2y);
         }
         if (m < 0) {                       
                 m=0;  
              }

          cmi[x1][x2] = m;
          cmi[x2][x1] = m;    
      }     
    }
  }
    
     
    // find the maximum spanning tree

	CategoricalAttribute firstAtt = 0;

	parents_[firstAtt] = NOPARENT;

	float *maxWeight;
	CategoricalAttribute *bestSoFar;
	CategoricalAttribute topCandidate = firstAtt;
	std::set<CategoricalAttribute> available;

	safeAlloc(maxWeight, noCatAtts_);
	safeAlloc(bestSoFar, noCatAtts_);

	maxWeight[firstAtt] = -std::numeric_limits<float>::max();

	for (CategoricalAttribute a = firstAtt + 1; a < noCatAtts_; a++) {
           if  (generalizationSet[a]==false){
		maxWeight[a] = cmi[firstAtt][a];
		if (cmi[firstAtt][a] > maxWeight[topCandidate])
			topCandidate = a;
		bestSoFar[a] = firstAtt;
		available.insert(a);
           }     
	}

	while (!available.empty()) {
		const CategoricalAttribute current = topCandidate;
		parents_[current] = bestSoFar[current];
		available.erase(current);

		if (!available.empty()) {
			topCandidate = *available.begin();
			for (std::set<CategoricalAttribute>::const_iterator it =
					available.begin(); it != available.end(); it++) {
				if (maxWeight[*it] < cmi[current][*it]) {
					maxWeight[*it] = cmi[current][*it];
					bestSoFar[*it] = current;
				}

				if (maxWeight[*it] > maxWeight[topCandidate])
					topCandidate = *it;
			}
		}
	}


	delete[] bestSoFar;
	delete[] maxWeight;


    
	for (CatValue y = 0; y < noClasses_; y++) {
		classDist[y] = xxyDist_.xyCounts.p(y);
	}

	for (unsigned int x1 = 0; x1 < noCatAtts_ ; x1++) {
		const CategoricalAttribute parent = parents_[x1];
  
		  if (parent == NOPARENT) {
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);
			}
		  } else {
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] *= xxyDist_.p(x1, inst.getCatVal(x1), parent,
						inst.getCatVal(parent), y);
			}

             }
	}

	normalise(classDist);
}

void TAN_gen::finalisePass() {
	
        trainingIsFinished_=true;
	
}

/// true iff no more passes are required. updated by finalisePass()
bool TAN_gen::trainingIsFinished() {
	return trainingIsFinished_;
}
