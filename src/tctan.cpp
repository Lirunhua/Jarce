/* 
 * File:   tctan.cpp
 * Author: Administrator
 * 
 * Created on 2016年6月13日, 下午1:56
 */

#include "tctan.h"
#include "utils.h"
#include "correlationMeasures.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>

tctan::tctan() :
trainingIsFinished_(false) {
}

tctan::tctan(char* const *&, char* const *) :
xxyDist_(), trainingIsFinished_(false) {
    name_ = "tctan";
}

tctan::~tctan(void) {
}

void tctan::reset(InstanceStream &is) {
     //printf("reset\n");
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    trainingIsFinished_ = false;

    //safeAlloc(parents, noCatAtts_);
    parents_.resize(noClasses_);
    for (CategoricalAttribute a = 0; a < noClasses_; a++) {
        parents_[a].resize(noCatAtts_);
    }
    for (CategoricalAttribute a = 0; a < noClasses_; a++) {
        for (CategoricalAttribute b = 0; b < noCatAtts_; b++) {
            parents_[a][b] = NOPARENT;
        }
    }

    xxyDist_.reset(is);
}

void tctan::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void tctan::initialisePass() {
    assert(trainingIsFinished_ == false);
    //learner::initialisePass (pass_);
    //	dist->clear();
    //	for (CategoricalAttribute a = 0; a < meta->noCatAtts; a++) {
    //		parents_[a] = NOPARENT;
    //	}
}

void tctan::train(const instance &inst) {
    xxyDist_.update(inst);
}

void tctan::classify(const instance &inst, std::vector<double> &classDist) {
    //printf("classify\n");
    crosstab<double> posteriorDist_ = crosstab<double>(noClasses_);
    for (unsigned int c = 0; c < noClasses_; c++) {
        posteriorDist_[c].assign(noClasses_, 0);
    }
    
    double maxp = 0.0;
    CatValue select = 0;
    
    for (CategoricalAttribute c = 0; c < noClasses_; c++) {

        for (CatValue y = 0; y < noClasses_; y++) {
            posteriorDist_[c][y] = xxyDist_.xyCounts.p(y);
        }

        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
            const CategoricalAttribute parent = parents_[c][x1];

            if (parent == NOPARENT) {
                for (CatValue y = 0; y < noClasses_; y++) {
                    posteriorDist_[c][y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);
                }
            } else {
                for (CatValue y = 0; y < noClasses_; y++) {
                    posteriorDist_[c][y] *= xxyDist_.p(x1, inst.getCatVal(x1), parent,
                            inst.getCatVal(parent), y);
                }
            }
        }
        //选出联合概率最大的一组classDist
        for (CatValue y = 0; y < noClasses_; y++) {
            if (maxp < posteriorDist_[c][y]) {
                maxp = posteriorDist_[c][y];
                select = c;
            }
        }
    }
    
    for (CatValue y = 0; y < noClasses_; y++) {
        classDist[y] = posteriorDist_[select][y];
    }
    
    normalise(classDist);
}

void tctan::finalisePass() {
    //printf("finalisePass\n");
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

    int maxitem = 0; //解决栈溢出
        if (noCatAtts_ > noClasses_)
            maxitem = noCatAtts_;
        else
            maxitem = noClasses_;
    crosstab3D<double> cmi = crosstab3D<double>(maxitem);

    getCondMutualInfTC(xxyDist_, cmi);

    // find the maximum spanning tree
    for (CategoricalAttribute c = 0; c < noClasses_; c++) {

        CategoricalAttribute firstAtt = 0;

        parents_[c][firstAtt] = NOPARENT;

        float *maxWeight;
        CategoricalAttribute *bestSoFar;
        CategoricalAttribute topCandidate = firstAtt;
        std::set<CategoricalAttribute> available;

        safeAlloc(maxWeight, noCatAtts_);
        safeAlloc(bestSoFar, noCatAtts_);

        maxWeight[firstAtt] = -std::numeric_limits<float>::max();

        for (CategoricalAttribute a = firstAtt + 1; a < noCatAtts_; a++) {
            maxWeight[a] = cmi[c][firstAtt][a];
            if (cmi[c][firstAtt][a] > maxWeight[topCandidate])
                topCandidate = a;
            bestSoFar[a] = firstAtt;
            available.insert(a);
        }

        while (!available.empty()) {
            const CategoricalAttribute current = topCandidate;
            parents_[c][current] = bestSoFar[current];
            available.erase(current);

            if (!available.empty()) {
                topCandidate = *available.begin();
                for (std::set<CategoricalAttribute>::const_iterator it =
                        available.begin(); it != available.end(); it++) {
                    if (maxWeight[*it] < cmi[c][current][*it]) {
                        maxWeight[*it] = cmi[c][current][*it];
                        bestSoFar[*it] = current;
                    }

                    if (maxWeight[*it] > maxWeight[topCandidate])
                        topCandidate = *it;
                }
            }
        }
        
        delete[] bestSoFar;
        delete[] maxWeight;
    }
    //for (attribute a = 0; a < meta->noAttributes; a++) {
    //  delete []mi[a];
    //}
    //delete []mi;
    

    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool tctan::trainingIsFinished() {
    return trainingIsFinished_;
}
