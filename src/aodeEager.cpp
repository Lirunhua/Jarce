/* Petal: An open source system for classification learning from very large data
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
 ** Please report any bugs to Shenglei Chen <tristan_chen@126.com>
 */

#include "aodeEager.h"
#include <assert.h>
#include "utils.h"
#include <algorithm>
#include "correlationMeasures.h"
#include "globals.h"
#include "utils.h"

aodeEager::aodeEager(char* const *& argv, char* const * end) {
	name_ = "aodeEager";

	weightedMI = false;
	weightedSU = false;
	minCount = 100;
	subsumptionResolution = false;

//else if (argv[0][1] == 't') {
//      if (streq(argv[0] + 2, "LSR"))
//        srt = srtLSR;
//      else if (streq(argv[0] + 2, "ESR"))
//        srt = srtESR;
//      else if (streq(argv[0] + 2, "BSE"))
//        srt = srtBSE;
//      else if (streq(argv[0] + 2, "NSR"))
//        srt = srtNSR;
//      else
//        error("aodeEager does not support argument %s", argv[0]);
//    }

// get arguments
	while (argv != end) {
		if (*argv[0] != '-') {
			break;
		} else if (streq(argv[0] + 1, "subsumption")) {
			subsumptionResolution = true;
		} else if (argv[0][1] == 'c') {
			getUIntFromStr(argv[0] + 2, minCount, "c");

		} else if (streq(argv[0] + 1, "wmi")) {
			weightedMI = true;
		} else if (streq(argv[0] + 1, "wsu")) {
			weightedSU = true;
		}

		else {
			break;
		}

		name_ += *argv;

		++argv;
	}

	trainingIsFinished_ = false;
}

aodeEager::~aodeEager(void) {

}

void aodeEager::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void aodeEager::reset(InstanceStream &is) {
	xxyDist_.reset(is);
	trainingIsFinished_ = false;

	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();

	generalizationSet.resize(noCatAtts_, -1);
	substitutionSet.resize(noCatAtts_, -1);
	weight.resize(noCatAtts_, 1);


}

void aodeEager::initialisePass() {

}

void aodeEager::train(const instance &inst) {
	xxyDist_.update(inst);
}

/// true iff no more passes are required. updated by finalisePass()
bool aodeEager::trainingIsFinished() {
	return trainingIsFinished_;
}

void aodeEager::classify(const instance &inst, std::vector<double> &classDist) {

	generalizationSet.assign(noCatAtts_, -1);
	substitutionSet.assign(noCatAtts_, -1);

	//compute the generalisation set and substitution set for
	//lazy subsumption resolution
	if (subsumptionResolution == true) {

		for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
			if (substitutionSet[i] == -1) {

				InstanceCount countOfxi = xxyDist_.xyCounts.getCount(i,
						inst.getCatVal(i));

				for (CategoricalAttribute j = 0; j < noCatAtts_; j++) {
					if (j != i) {
						InstanceCount countOfxixj = xxyDist_.getCount(i,
								inst.getCatVal(i), j, inst.getCatVal(j));
						InstanceCount countOfxj = xxyDist_.xyCounts.getCount(j,
								inst.getCatVal(j));

						if (countOfxj == countOfxixj && countOfxj >= minCount) {

							//xi is a substitution of xj
							// record xi is a substitution of xj so that xj can be neglect in rest loop
							//as i increases from 0, so xi is kept as the min(Si)
							if (countOfxi == countOfxj)
								substitutionSet[j] = i;
							//xi is a generalisation of xj
							//once one xj has been found for xi, stop for rest j
							else {
								generalizationSet[i] = j;
								break;
							}
						}
					}
				}
			}
		}

		if (verbosity >= 5) {
			for (CategoricalAttribute i = 0; i < noCatAtts_; i++)
				if (generalizationSet[i] == -1 && substitutionSet[i] == -1)
					printf("%d\t", i);
			printf("\n");
		}
	}

	if (verbosity >= 5) {
		for (CatValue i = 0; i < noCatAtts_; i++) {
			printf("%f\n", weight[i]);
		}
	}

	for (CatValue y = 0; y < noClasses_; y++)
		classDist[y] = 0;

	// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max() / noCatAtts_;

	fdarray<double> spodeProbs(noCatAtts_,noClasses_);

	CatValue delta = 0;

	for (CatValue parent = 0; parent < noCatAtts_; parent++) {

		//discard the attribute that in in generalization set or substitute set
		if (generalizationSet[parent] == -1 && substitutionSet[parent] == -1) {

			CatValue parentVal = inst.getCatVal(parent);

			if (xxyDist_.xyCounts.getCount(parent, parentVal) > 0) {

				delta++;
				for (CatValue y = 0; y < noClasses_; y++) {

					spodeProbs[parent][y] = weight[parent]
							* xxyDist_.xyCounts.jointP(parent, parentVal, y)
							* scaleFactor;
				}

			} else if (verbosity >= 5)
				printf("%d\n", parent);

		}
	}

	if (delta == 0) {
		nbClassify(inst, classDist, xxyDist_.xyCounts);
		return;
	}

	for (CategoricalAttribute parent = 1; parent < noCatAtts_; parent++) {

		CatValue parentVal = inst.getCatVal(parent);

		std::vector<std::vector<std::vector<double> > > * parentsProbs =
				&xxyDist_.condiProbs[parent][parentVal];

		//discard the attribute that in in generalization set or substitute set
		if (generalizationSet[parent] == -1 && substitutionSet[parent] == -1) {

			//filter out the parent that does not appear
			if (xxyDist_.xyCounts.getCount(parent, parentVal) == 0)
				continue;

			//delta++;

			for (CategoricalAttribute child = 0; child < parent; child++) {

				//	printf("c:%d\n", child);
				if (generalizationSet[child] == -1
						&& substitutionSet[child] == -1) {
					if (child != parent) {

						CatValue childVal = inst.getCatVal(child);
						std::vector<double> *childProbs =
								&xxyDist_.condiProbs[child][childVal][parent][parentVal];

						std::vector<double> *parentChildProbs =
								&(*parentsProbs)[child][childVal];

						for (CatValue y = 0; y < noClasses_; y++) {

							spodeProbs[parent][y] *= (*childProbs)[y];
							spodeProbs[child][y] *= (*parentChildProbs)[y];
						}
					}
				}

			}

		}
	}

	for (CatValue parent = 0; parent < noCatAtts_; parent++) {
		//discard the attribute that in in generalization set or substitute set
		if (generalizationSet[parent] == -1 && substitutionSet[parent] == -1) {
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] += spodeProbs[parent][y];
			}
		}
	}

	normalise(classDist);

}
void aodeEager::finalisePass() {

	trainingIsFinished_ = true;

//	if (weighted) {
//		weight.assign(noCatAtts_, 0);
//		getMutualInformation(xxyDist_.xyCounts, weight);
//
//		//calculate the mutual information using count of instance rather than  m-esitmated probability
//		if (verbosity >= 4) {
//			for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
//				printf("%f\n", weight[a]);
//			}
//		}
//	}

//calculate the mutual information using count of instance rather than  m-esitmated probability
//
//	if (weighted) {
//
//		weight.assign(noCatAtts_, 0);
//
//		for (CatValue i = 0; i < noCatAtts_; i++) {
//
//			for (CatValue x = 0; x <xxyDist_.instanceStream_->getNoValues(i); x++) {
//
//				for (CatValue y = 0; y <noClasses_; y++) {
//
//					InstanceCount countXy = xxyDist_.xyCounts.getCount(i,
//							x,y);
//					InstanceCount countY = xxyDist_.xyCounts.getClassCount(y);
//					InstanceCount countX = xxyDist_.xyCounts.getCount(i,x);
//					InstanceCount count = xxyDist_.xyCounts.count;
//
//
//					if(countXy==0)
//						continue;
//					double weightXy=(double)countXy / count
//							* log2(((double)countXy / countY) * ((double)count / countX));
//					weight[i] +=weightXy ;
//				}
//			}
//			//printf("%f\n",weight[i]);
//		}
//		//normalise(weight);
//	}

//calculate the mutual information using m-esitmated probability

	if (weightedMI) {

		weight.assign(noCatAtts_, 0);

		for (CatValue i = 0; i < noCatAtts_; i++) {

			for (CatValue x = 0; x < xxyDist_.instanceStream_->getNoValues(i);
					x++) {
				for (CatValue y = 0; y < noClasses_; y++) {
					double pXy = xxyDist_.xyCounts.jointP(i, x, y);
					double pY = xxyDist_.xyCounts.p(y);

					double pX = 0;
					for (CatValue yPrime = 0; yPrime < noClasses_; yPrime++) {
						pX += xxyDist_.xyCounts.jointP(i, x, yPrime);
					}

					if (pXy == 0)
						continue;
					double weightXy = pXy * log2(pXy / (pX * pY));
					weight[i] += weightXy;
				}
			}
		}
	} else if (weightedSU) {

		weight.assign(noCatAtts_, 0);

		//calculate the entropy of class
		double entropyY = 0;
		for (CatValue y = 0; y < noClasses_; y++) {

			double pY = xxyDist_.xyCounts.p(y);
			if (pY == 0)
				continue;
			entropyY -= pY * log2(pY);
		}

		//calculate the entropy of each attribute
		std::vector<double> entropyX(noCatAtts_, 0);

		for (CatValue i = 0; i < noCatAtts_; i++) {
			for (CatValue x = 0; x < xxyDist_.instanceStream_->getNoValues(i);
					x++) {
				double pX = 0;
				for (CatValue yPrime = 0; yPrime < noClasses_; yPrime++) {
					pX += xxyDist_.xyCounts.jointP(i, x, yPrime);
				}
				if (pX == 0)
					continue;
				entropyX[i] -= pX * log2(pX);
			}
		}

		//calculate the mutual information between each attribute and class
		std::vector<double> mutualInfo(noCatAtts_, 0);

		for (CatValue i = 0; i < noCatAtts_; i++) {
			for (CatValue x = 0; x < xxyDist_.instanceStream_->getNoValues(i);
					x++) {

				for (CatValue y = 0; y < noClasses_; y++) {
					double pXy = xxyDist_.xyCounts.jointP(i, x, y);
					double pY = xxyDist_.xyCounts.p(y);

					double pX = 0;
					for (CatValue yPrime = 0; yPrime < noClasses_; yPrime++) {
						pX += xxyDist_.xyCounts.jointP(i, x, yPrime);
					}
					if (pXy == 0)
						continue;
					double weightXy = pXy * log2(pXy / (pX * pY));

					mutualInfo[i] += weightXy;
				}
			}
		}

		//calculate the symmetric uncertainty
		for (CatValue i = 0; i < noCatAtts_; i++) {
			weight[i] = 2 * mutualInfo[i] / (entropyX[i] + entropyY);
		}

	}

	if (verbosity >= 4) {
		for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
			printf("%f\n", weight[a]);
		}
	}

	xxyDist_.calculateCondProb();

}

void aodeEager::nbClassify(const instance &inst, std::vector<double> &classDist,
		xyDist &xyDist_) {

	for (CatValue y = 0; y < noClasses_; y++) {
		double p = xyDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
		// scale up by maximum possible factor to reduce risk of numeric underflow

		for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
			p *= xyDist_.p(a, inst.getCatVal(a), y);
		}

		assert(p >= 0.0);
		classDist[y] = p;
	}
	normalise(classDist);
}

