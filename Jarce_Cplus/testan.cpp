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
#include "testan.h"
#include "utils.h"
#include "correlationMeasures.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>

testan::testan() :
trainingIsFinished_(false) {
}

testan::testan(char* const *&, char* const *) :
xxyDist_(), trainingIsFinished_(false) {
    name_ = "testan";
}

testan::~testan(void) {
}

void testan::reset(InstanceStream &is) {
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
    parents_1.resize(noCatAtts);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_1[a] = NOPARENT;
    }

    xxyDist_.reset(is);
    xxxyDist_.reset(is);
}

void testan::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void testan::initialisePass() {
    assert(trainingIsFinished_ == false);
    //learner::initialisePass (pass_);
    //	dist->clear();
    //	for (CategoricalAttribute a = 0; a < meta->noCatAtts; a++) {
    //		parents_[a] = NOPARENT;
    //	}
}

void testan::train(const instance &inst) {
    xxyDist_.update(inst);
    xxxyDist_.update(inst);
}

void testan::classify(const instance &inst, std::vector<double> &classDist) {

    for (CatValue y = 0; y < noClasses_; y++) {
        classDist[y] = xxyDist_.xyCounts.p(y);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
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

void testan::finalisePass() {
    assert(trainingIsFinished_ == false);

    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    getCondMutualInf(xxyDist_, cmi);

    // find the maximum spanning tree

    CategoricalAttribute firstAtt = 0;

    parents_[firstAtt] = NOPARENT;

    float *maxWeight;
    CategoricalAttribute *bestSoFar;
    CategoricalAttribute topCandidate = firstAtt;
    std::set<CategoricalAttribute> available;
    //..........
    std::set<CategoricalAttribute> used;
    used.insert(firstAtt);

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

    parents_[topCandidate] = bestSoFar[topCandidate];
    available.erase(topCandidate);
    used.insert(topCandidate);
    printf("parents_:\n");
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        if (parents_[i] != NOPARENT)
            printf("%d->%d\n", parents_[i], i);
    }
    printf("**\n");

    double maxIxxx = -std::numeric_limits<float>::max();
    CategoricalAttribute pos_ch = 0xFFFFFFFFUL;
    CategoricalAttribute pos_fa = 0xFFFFFFFFUL;
    CategoricalAttribute pos = 0xFFFFFFFFUL;
    for (std::set<CategoricalAttribute>::const_iterator it0 = used.begin(); it0 != used.end(); it0++) {//父
        for (std::set<CategoricalAttribute>::const_iterator it = available.begin(); it != available.end(); it++) { //子
            for (unsigned int i = 0; i < noCatAtts_; i++) {
                parents_1[i] = parents_[i];
            }
            parents_1[*it] = *it0;
            double tempIxxx = 0;
            tempIxxx = getIxxx(xxxyDist_, firstAtt, topCandidate, *it, parents_1);
            for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                parents_1[a] = NOPARENT;
            }
            if (maxIxxx < tempIxxx) {
                maxIxxx = tempIxxx;
                pos = *it;
            }
        }
    }
    printf("fianlly choose %d\n", pos);


    //    while (!available.empty()) {
    //        const CategoricalAttribute current = topCandidate;
    //        parents_[current] = bestSoFar[current];
    //
    //        available.erase(current);
    //
    //        if (!available.empty()) {
    //            topCandidate = *available.begin();
    //            for (std::set<CategoricalAttribute>::const_iterator it =
    //                    available.begin(); it != available.end(); it++) {
    //                if (maxWeight[*it] < cmi[current][*it]) {
    //                    maxWeight[*it] = cmi[current][*it];
    //                    bestSoFar[*it] = current;
    //                }
    //
    //                if (maxWeight[*it] > maxWeight[topCandidate])
    //                    topCandidate = *it;
    //            }
    //        }
    //
    //    }

    delete[] bestSoFar;
    delete[] maxWeight;

    trainingIsFinished_ = true;
    printf("-----------------\n");






}

/// true iff no more passes are required. updated by finalisePass()

bool testan::trainingIsFinished() {
    return trainingIsFinished_;
}
