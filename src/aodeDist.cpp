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

#include "aodeDist.h"
#include <assert.h>
#include "utils.h"
#include <algorithm>
#include "correlationMeasures.h"
#include "globals.h"

aodeDist::aodeDist(char* const *& argv, char* const * end) {
	name_ = "AODE";

	weighted = false;
	minCount = 100;
	subsumptionResolution = false;

// get arguments
	while (argv != end) {
		if (*argv[0] != '-') {
			break;
		} else if (streq(argv[0] + 1, "subsumption")) {
			subsumptionResolution = true;
		} else if (argv[0][1] == 'c') {
			getUIntFromStr(argv[0] + 2, minCount, "c");
		} else if (streq(argv[0] + 1, "weighted")) {
			weighted = true;
		} else {
			break;
		}

		name_ += *argv;

		++argv;
	}

	trainingIsFinished_ = false;
}

aodeDist::~aodeDist(void) {
}

void aodeDist::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void aodeDist::reset(InstanceStream &is) {
	//xxyDist_.reset(is);
	instanceStream_ = &is;

	trainingIsFinished_ = false;

	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();

	generalizationSet.resize(noCatAtts_, -1);
	substitutionSet.resize(noCatAtts_, -1);
	weight.resize(noCatAtts_, 1);

	// set the count number
	count = 0;

	//initialise the class counts
	classCounts.assign(noClasses_, 0);

	//initialise the one-d offset for xyCounts
	attOffset.resize(noCatAtts_);
	unsigned int next = 0;
	for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
		attOffset[a] = next;
		next += is.getNoValues(a);
	}

	// initialise the xyCounts
	xyCounts_.resize(next);
	for (CatValue c = 0; c < next; c++) {
		xyCounts_[c].assign(noClasses_, 0);
	}

	//initialise the xxyCounts
	xxyCounts_.resize(noCatAtts_);
	for (CategoricalAttribute i = 1; i < noCatAtts_; i++) {

		CatValue firstNoValues = is.getNoValues(i);

		xxyCounts_[i].resize(firstNoValues);

		for (CatValue j = 0; j < firstNoValues; j++) {
			xxyCounts_[i][j].resize(i);

			for (CategoricalAttribute k = 0; k < i; k++) {

				CatValue secondNoValues = is.getNoValues(k);

				xxyCounts_[i][j][k].resize(secondNoValues);

				for (CatValue l = 0; l < secondNoValues; l++) {

					xxyCounts_[i][j][k][l].assign(noClasses_,0);
				}
			}
		}
	}

	//initialise the conditional probability
	condiProbs.resize(noCatAtts_);
	for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {

		CatValue firstNoValues = is.getNoValues(i);

		condiProbs[i].resize(firstNoValues);

		for (CatValue j = 0; j < firstNoValues; j++) {
			condiProbs[i][j].resize(noCatAtts_);

			for (CategoricalAttribute k = 0; k < noCatAtts_; k++) {

				CatValue secondNoValues = is.getNoValues(k);

				condiProbs[i][j][k].resize(secondNoValues);

				for (CatValue l = 0; l < secondNoValues; l++) {

					condiProbs[i][j][k][l].assign(noClasses_, 1.0);
				}
			}
		}
	}
}

void aodeDist::initialisePass() {

}

void aodeDist::train(const instance &i) {

	CatValue theClass = i.getClass();

	//update the count of instance
	count++;

	//udpate the count of each class
	classCounts[theClass]++;

	//update the count for the current attribute value and class

	for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
		xyCounts_[attOffset[a] + i.getCatVal(a)][theClass]++;
	}

	//update the count of each two attrbitue and class

	for (CategoricalAttribute x1 = 1; x1 < instanceStream_->getNoCatAtts();
			x1++) {
		CatValue v1 = i.getCatVal(x1);

		std::vector<std::vector<std::vector<InstanceCount> > > *countsForAtt =
				&xxyCounts_[x1][v1];

		for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

			CatValue v2 = i.getCatVal(x2);

			(*countsForAtt)[x2][v2][theClass]++;

			//	assert(count[x1][v1][x2][v2 * noClasses_ + theClass] <= count);
		}
	}

}

/// true iff no more passes are required. updated by finalisePass()
bool aodeDist::trainingIsFinished() {
	return trainingIsFinished_;
}

void aodeDist::classify(const instance &inst, std::vector<double> &classDist) {

	generalizationSet.assign(noCatAtts_, -1);
	substitutionSet.assign(noCatAtts_, -1);

	//compute the generalisation set and substitution set for
	//lazy subsumption resolution
	if (subsumptionResolution == true) {

		for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
			if (substitutionSet[i] == -1) {
				InstanceCount countOfxi = getCount(i, inst.getCatVal(i));

				for (CategoricalAttribute j = 0; j < noCatAtts_; j++) {
					if (j != i) {
						InstanceCount countOfxixj = getCount(i,
								inst.getCatVal(i), j, inst.getCatVal(j));
						InstanceCount countOfxj = getCount(j,
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

		if (verbosity >= 4) {
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

	CatValue delta = 0;

	fdarray<double> spodeProbs(noCatAtts_,noClasses_);

	for (CatValue parent = 0; parent < noCatAtts_; parent++) {

		//discard the attribute that in in generalization set or substitute set
		if (generalizationSet[parent] == -1 && substitutionSet[parent] == -1) {

			CatValue parentVal = inst.getCatVal(parent);

			if (getCount(parent, parentVal) > 0) {

				delta++;
				for (CatValue y = 0; y < noClasses_; y++) {

					spodeProbs[parent][y] = weight[parent]
							* jointP(parent, parentVal, y) * scaleFactor;
				}

			} else if (verbosity >= 5)
				printf("%d\n", parent);

		}
	}

	if (delta == 0) {
		nbClassify(inst, classDist);
		return;
	}

	for (CategoricalAttribute parent = 1; parent < noCatAtts_; parent++) {

		CatValue parentVal = inst.getCatVal(parent);

		std::vector<std::vector<std::vector<double> > > * parentsProbs =
				&condiProbs[parent][parentVal];

		//discard the attribute that in in generalization set or substitute set
		if (generalizationSet[parent] == -1 && substitutionSet[parent] == -1) {

			//filter out the parent that does not appear
			if (getCount(parent, parentVal) == 0)
				continue;

			//delta++;

			for (CategoricalAttribute child = 0; child < parent; child++) {

				//	printf("c:%d\n", child);
				if (generalizationSet[child] == -1
						&& substitutionSet[child] == -1) {
					if (child != parent) {

						CatValue childVal = inst.getCatVal(child);
						std::vector<double> *childProbs =
								&condiProbs[child][childVal][parent][parentVal];

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
void aodeDist::finalisePass() {

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

	if (weighted) {

		weight.assign(noCatAtts_, 0);

		for (CatValue i = 0; i < noCatAtts_; i++) {

			for (CatValue x = 0; x < instanceStream_->getNoValues(i); x++) {
				for (CatValue y = 0; y < noClasses_; y++) {
					double pXy = jointP(i, x, y);
					double pY = p(y);

					double pX = 0;
					for (CatValue yPrime = 0; yPrime < noClasses_; yPrime++) {
						pX += jointP(i, x, yPrime);
					}

					if (pXy == 0)
						continue;
					double weightXy = pXy * log2(pXy / (pX * pY));
					weight[i] += weightXy;
				}
			}
		}
	}

	if (verbosity >= 4) {
		for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
			printf("%f\n", weight[a]);
		}
	}

	//compute the conditional probability

	for (CategoricalAttribute i = 1; i < noCatAtts_; i++) {
		CatValue firstNoValues = instanceStream_->getNoValues(i);

		for (CatValue j = 0; j < firstNoValues; j++) {

			std::vector<InstanceCount> *parentVecotr = &xyCounts_[attOffset[i]
					+ j];
			std::vector<std::vector<std::vector<InstanceCount> > > *parentVecotr2 =
					&xxyCounts_[i][j];

			for (CategoricalAttribute k = 0; k < i; k++) {
				CatValue secondNoValues = instanceStream_->getNoValues(k);
				for (CatValue l = 0; l < secondNoValues; l++) {

					std::vector<InstanceCount> *childVector =
							&xyCounts_[attOffset[k] + l];
					std::vector<InstanceCount> *childVector2 =
							&(*parentVecotr2)[k][l];

					for (CatValue y = 0; y < noClasses_; y++) {

						double parentYM = (*parentVecotr)[y] + M;

						double childYM = (*childVector)[y] + M;

						InstanceCount parentChildY = (*childVector2)[y];

						condiProbs[i][j][k][l][y] = (parentChildY
								+ M / firstNoValues) / childYM;
						condiProbs[k][l][i][j][y] = (parentChildY
								+ M / secondNoValues) / parentYM;

					}

				}
			}
		}
	}

}

void aodeDist::nbClassify(const instance &inst,
		std::vector<double> &classDist) {

	for (CatValue y = 0; y < noClasses_; y++) {
		double prob = p(y) * (std::numeric_limits<double>::max() / 2.0);
		// scale up by maximum possible factor to reduce risk of numeric underflow

		for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
			prob *= p(a, inst.getCatVal(a), y);
		}

		assert(prob >= 0.0);
		classDist[y] = prob;
	}
	normalise(classDist);
}

