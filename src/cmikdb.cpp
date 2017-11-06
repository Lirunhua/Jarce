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
 ** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "cmikdb.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

cmikdb::cmikdb() : pass_(1) {
}

cmikdb::cmikdb(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "cmikdb";

    // defaults
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

cmikdb::~cmikdb(void) {
}

void cmikdb::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

// creates a comparator for two attributes based on their relative mutual information with the class

class miCmpClass {
public:

    miCmpClass(std::vector<float> *m) {
        mi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
        return (*mi)[a] > (*mi)[b];
    }

private:
    std::vector<float> *mi;
};

void cmikdb::reset(InstanceStream &is) {
    //printf("reset\n");
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1

    // initialise distributions
    //dTree_.resize(noCatAtts);
    parents_.resize(noCatAtts);
    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        parents_[a].resize(noCatAtts);
    }
    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        for (CategoricalAttribute b = 0; b < noCatAtts; b++) {
            parents_[a][b].clear();
        }
    }

    /*初始化各数据结构空间*/
    dist_.reset(is); //

    classDist_.reset(is);
    trainingIsFinished_ = false;
    //pass_ = 1;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void cmikdb::train(const instance &inst) {
    //printf("train\n");
    /*if (pass_ == 1) {
        dist_.update(inst);
    }
    else {
        assert(pass_ == 2);
        classDist_.update(inst);
    }*/
    dist_.update(inst);
    classDist_.update(inst);
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void cmikdb::initialisePass() {
    assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void cmikdb::finalisePass() {
    //if (pass_ == 1 && k_!=0) {
    assert(trainingIsFinished_ == false);
    printf("finalisePass\n");

    // calculate the conditional mutual information from the xxy distribution
    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    getCondMutualInf(dist_.xxyCounts, cmi);

    //dist_.clear();


    //printf("\nConditional mutual information table\n");
    // cmi.print();

    std::vector<CategoricalAttribute> order;
    //order.resize(noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        order.push_back(a);
    }
    for (CatValue ai = 0; ai < noCatAtts_; ai++) {
        for (std::vector<CategoricalAttribute>::const_iterator it = order.begin(); it != order.end(); it++) {//做一次遍历打印
            printf("%d\t", *it);
        }
        printf("\n");
        // assign the parents
        if (!order.empty()) {
            //miCmpClass cmp(&mi);

            //std::sort(order.begin(), order.end(), cmp);

            if (verbosity >= 2) {
                printf("\n%s parents:\n", instanceStream_->getCatAttName(order[0]));
            }

            // proper KDB assignment of parents
            for (std::vector<CategoricalAttribute>::const_iterator it = order.begin() + 1; it != order.end(); it++) {
                parents_[ai][*it].push_back(order[0]);
                for (std::vector<CategoricalAttribute>::const_iterator it2 = order.begin() + 1; it2 != it; it2++) {
                    // make parents into the top k attributes on mi that precede *it in order
                    if (parents_[ai][*it].size() < k_) {
                        // create space for another parent
                        // set it initially to the new parent.
                        // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
                        parents_[ai][*it].push_back(*it2);
                    }
                    for (unsigned int i = 0; i < parents_[ai][*it].size(); i++) {
                        if (cmi[*it2][*it] > cmi[parents_[ai][*it][i]][*it]) {
                            // move lower value parents down in order
                            for (unsigned int j = parents_[ai][*it].size() - 1; j > i; j--) {
                                parents_[ai][*it][j] = parents_[ai][*it][j - 1];
                            }
                            // insert the new att
                            parents_[ai][*it][i] = *it2;
                            break;
                        }
                    }
                }
                printf("%d parents: ", *it);
                for (unsigned int i = 0; i < parents_[ai][*it].size(); i++) {
                    printf("%d ", parents_[ai][*it][i]);
                }
                putchar('\n');
            }

        }
        CategoricalAttribute temp = *order.begin();
        order.erase(order.begin());
        order.push_back(temp);
        /*for(std::vector<CategoricalAttribute>::const_iterator it = order.begin(); it != order.end(); it++){//做一次遍历打印
                printf("%d/t",*it);
        }
        printf("!!!!!!!!!\n");*/
    }
    //}

    trainingIsFinished_ = true;
    //++pass_;
}

/// true iff no more passes are required. updated by finalisePass()

bool cmikdb::trainingIsFinished() {
    return trainingIsFinished_;
    // return pass_ > 2;
}

void cmikdb::classify(const instance& inst, std::vector<double> &posteriorDist) {
    // calculate the class probabilities in parallel
    // P(y)
    //printf("classify\n");
    
    std::vector<std::vector<double> > p;
    for (CatValue ai = 0; ai < noCatAtts_; ai++) {
        p.resize(noCatAtts_);
        p[ai].assign(noClasses_,0);
    }
    
    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
    }
    for (CatValue ai = 0; ai < noCatAtts_; ai++) {
        //CatValue ai = 0;
        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {

            if (parents_[ai][x1].size() == 0) {
                //printf("PARent=0  \n");
                for (CatValue y = 0; y < noClasses_; y++) {
                    posteriorDist[y] *= dist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate

                }
            } else if (parents_[ai][x1].size() == 1) {
                //printf("PARent=1  \n");
                for (CatValue y = 0; y < noClasses_; y++) {
                    posteriorDist[y] *= dist_.xxyCounts.p(x1, inst.getCatVal(x1), parents_[ai][x1][0], inst.getCatVal(parents_[ai][x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate


                }
            } else if (parents_[ai][x1].size() == 2) {
                //printf("PARent=2  \n");
                for (CatValue y = 0; y < noClasses_; y++) {// p(x1=v1|Y=y, x2=v2, x3=v3) using M-estimate
                    posteriorDist[y] *= dist_.p(x1, inst.getCatVal(x1), parents_[ai][x1][0], inst.getCatVal(parents_[ai][x1][0]), parents_[ai][x1][1], inst.getCatVal(parents_[ai][x1][1]), y);

                }
            }
        }
        
        for (CatValue y = 0; y < noClasses_; y++) {
            p[ai][y] = posteriorDist[y];
        }
    }
    posteriorDist.assign(noClasses_,0);
    for (CatValue y = 0; y < noClasses_; y++) {
        for (CatValue ai = 0; ai < noCatAtts_; ai++) {
            posteriorDist[y] += p[ai][y];
        }
        posteriorDist[y] = posteriorDist[y]/noCatAtts_;
    }
    
    // normalise the results
    normalise(posteriorDist);
}



