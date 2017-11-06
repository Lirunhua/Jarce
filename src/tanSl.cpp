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
#include "tanSl.h"
#include "utils.h"
#include "correlationMeasures.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>

TANSl::TANSl() :
		trainingIsFinished_(false) {
}

TANSl::TANSl(char* const *& argv, char* const * end) :
		xxyDist_(), trainingIsFinished_(false) {
	name_ = "TANSl";
}

TANSl::~TANSl(void) {
	delete[] parents;
}

void TANSl::reset(InstanceStream &is) {
	instanceStream_ = &is;
	const unsigned int noCatAtts = is.getNoCatAtts();
	noCatAtts_ = noCatAtts;
	noClasses_ = is.getNoClasses();

	trainingIsFinished_ = false;

	safeAlloc(parents, noCatAtts_);
	for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
		parents[a] = NOPARENT;
	}

	xxyDist_.reset(is);
}

void TANSl::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void TANSl::initialisePass() {
	assert(trainingIsFinished_ == false);
	//learner::initialisePass (pass_);
//	dist->clear();
//	for (CategoricalAttribute a = 0; a < meta->noCatAtts; a++) {
//		parents[a] = NOPARENT;
//	}
}

void TANSl::train(const instance &inst) {
	xxyDist_.update(inst);
}

void TANSl::classify(const instance &inst, std::vector<double> &classDist) {

	for (CatValue y = 0; y < noClasses_; y++) {
		classDist[y] = xxyDist_.xyCounts.p(y);
	}

	for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
		const CategoricalAttribute parent = parents[x1];

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

void TANSl::finalisePass() {
	assert(trainingIsFinished_ == false);

	//// calculate conditional mutual information
	//float **mi = new float *[meta->noAttributes];

	//for (attribute a = 0; a < meta->noAttributes; a++) {
	//  mi[a] = new float[meta->noAttributes];
	//}

	//const double totalCount = dist->xyCounts.count;

	//for (attribute x1 = 1; x1 < meta->noAttributes; x1++) {
	//  if (meta->attTypes[x1] == categorical) {
	//    for (attribute x2 = 0; x2 < x1; x2++) {
	//      if (meta->attTypes[x2] == categorical) {
	//        float m = 0.0;

	//        for (cat_value v1 = 0; v1 < meta->noValues[x1]; v1++) {
	//          for (cat_value v2 = 0; v2 < meta->noValues[x2]; v2++) {
	//            for (unsigned int y = 0; y < meta->noClasses(); y++) {
	//              const double x1x2y = dist->getCount(x1, v1, x2, v2, y);
	//              if (x1x2y) {
	//                //const unsigned int yCount = dist->xyCounts.getClassCount(y);
	//                //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
	//                //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);
	//                m += (x1x2y/totalCount) * log(dist->xyCounts.getClassCount(y) * x1x2y / (static_cast<double>(dist->xyCounts.getCount(x1, v1, y))*dist->xyCounts.getCount(x2, v2, y)));
	//                //assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision
	//              }
	//            }
	//          }
	//        }

	//        assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision
	//        mi[x1][x2] = m;
	//        mi[x2][x1] = m;
	//      }
	//    }
	//  }
	//}

	crosstab<float> cmi = crosstab<float>(noCatAtts_);
	getCondMutualInf(xxyDist_, cmi);

	// find the maximum spanning tree

	CategoricalAttribute firstAtt = 0;

	parents[firstAtt] = NOPARENT;

	float *maxWeight;
	CategoricalAttribute *bestSoFar;
	CategoricalAttribute topCandidate = firstAtt;
	std::set<CategoricalAttribute> available;

	safeAlloc(maxWeight, noCatAtts_);
	safeAlloc(bestSoFar, noCatAtts_);

	maxWeight[firstAtt] = -std::numeric_limits<float>::max();

	for (CategoricalAttribute a = firstAtt + 1; a < noCatAtts_; a++) {
		maxWeight[a] = cmi[firstAtt][a];
		if (cmi[firstAtt][a] > maxWeight[topCandidate])
			topCandidate = a;
		bestSoFar[a] = firstAtt;
		available.insert(a);
	}

	while (!available.empty()) {
		const CategoricalAttribute current = topCandidate;
		parents[current] = bestSoFar[current];
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

	//for (attribute a = 0; a < meta->noAttributes; a++) {
	//  delete []mi[a];
	//}
	//delete []mi;
	delete[] bestSoFar;
	delete[] maxWeight;

	trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()
bool TANSl::trainingIsFinished() {
	return trainingIsFinished_;
}
