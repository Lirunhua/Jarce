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
 ** 这个算法类似于kdb,但是对每个属性的条件概率加权，这个想法借鉴于hidden naive bayes
 **
 ** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "kdbEager.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

kdbEager::kdbEager() : pass_(1) {
}

kdbEager::kdbEager(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "KDBEager";
    k_ = 1;


    // get arguments
    while (argv != end) {
        if (*argv[0] != '-') {
            break;
        } else if (argv[0][1] == 'k') {
            getUIntFromStr(argv[0] + 2, k_, "k");
        } else {
            break;
        }

        name_ += argv[0];

        ++argv;
    }

}

kdbEager::~kdbEager(void) {
}

void kdbEager::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
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

void kdbEager::reset(InstanceStream &is) {
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();
  
    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1

    // initialise distributions
    dTree_.resize(noCatAtts);
    parents_.resize(noCatAtts);

    SUM_Wij.resize(noCatAtts);
    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        Wij[a].resize(noCatAtts);
    }
    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        parents_[a].clear();
        dTree_[a].init(is, a);
    }

    dist_.reset(is);

    classDist_.reset(is);

    pass_ = 1;

}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

void kdbEager::train(const instance &inst) {
    if (pass_ == 1) {
        dist_.update(inst);
    } else {
        assert(pass_ == 2);

//        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
//            dTree_[a].update(inst, a, parents_[a]);
//        }
        classDist_.update(inst);
    }
}


/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void kdbEager::initialisePass() {
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void kdbEager::finalisePass() {
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
        getCondMutualInf(dist_, cmi);

        if (verbosity >= 3) {
            printf("\nConditional mutual information table\n");
            cmi.print();
        }

        // sort the attributes on MI with the class
       

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

            // proper kdbEager assignment of parents
            for (std::vector<CategoricalAttribute>::const_iterator it = order.begin() + 1; it != order.end(); it++) {
                parents_[*it].push_back(order[0]);
                for (std::vector<CategoricalAttribute>::const_iterator it2 = order.begin() + 1; it2 != it; it2++) {
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
                            for (unsigned int j = parents_[*it].size() - 1; j > i; j--) {
                                parents_[*it][j] = parents_[*it][j - 1];
                            }
                            // insert the new att
                            parents_[*it][i] = *it2;
                            break;
                        }
                    }
                }

                float SUM_cmi[noCatAtts_];
                for (std::vector<CategoricalAttribute>::const_iterator i = order.begin(); i != order.end(); i++) {
                    SUM_cmi[*i] = 0;
                    for (unsigned int j = 0; j < *i; j++) {
                        SUM_cmi[*i] += cmi[*i][j];
                    }                    
                    for (unsigned int j = 0; j < *i; j++) {
                        Wij[*i][j] = cmi[*i][j]/ SUM_cmi[*i];
                        float k=Wij[*i][j];
                        printf("the wij is %f",k);
                    }
                    printf("\n");
                }               
            }
        }
    } else if (pass_ == 2) {
        for (CategoricalAttribute x = 0; x < noCatAtts_; x++) {
            dTree_[x].calculateProbs(x);

        }
    }
    ++pass_;
}

/// true iff no more passes are required. updated by finalisePass()

bool kdbEager::trainingIsFinished() {
    return pass_ > 2;
}

void kdbEager::classify(const instance& inst, std::vector<double> &posteriorDist) {
    // calculate the class probabilities in parallel
    // P(y)

    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
    }

    // P(x_i | x_p1, .. x_pk, y)
//    for (CategoricalAttribute x = 0; x < noCatAtts_; x++) {
//        dTree_[x].updateClassDistribution(posteriorDist, x, inst);
//    }
    
// P(x_i | x_p1, .. x_pk, y)=Wi1*P(x_i | x_p1, y)+Wi2*P(x_i | x_p2, y)+.....+Wik*P(x_i | x_pk, y)
//    for (CatValue y = 0; y < noClasses_; y++) {
//        float PX[noCatAtts_];
//        for (std::vector<CategoricalAttribute>::const_iterator i = order.begin(); i != order.end(); i++) {
//            PX[i] = 0;           
//            for (unsigned int j = 0; j < parents_[i].size(); j++) {
//                const CatValue ival=inst.getCatVal(i);
//                const CatValue jval=inst.getCatVal(j);
//                PX[i] += Wij[i][j] * dist_.jointP(i,ival,j,jval,y)/dist_.xyCounts.jointP(j,jval,y);
//            }
//            posteriorDist[y] *= PX[i];
//        }
//    }
    // normalise the results
    normalise(posteriorDist);
}



