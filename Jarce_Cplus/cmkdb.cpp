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

#include "cmkdb.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

cmkdb::cmkdb() : pass_(1) {
}

cmkdb::cmkdb(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "cmkdb";

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

cmkdb::~cmkdb(void) {
}

void cmkdb::getCapabilities(capabilities &c) {
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

void cmkdb::reset(InstanceStream &is) {
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();
    cmilimit_ = noCatAtts_*(noCatAtts_-1)/2;
    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1
    result = 0.0;
    // initialise distributions
    dTree_.resize(noCatAtts);
    parents_.resize(noCatAtts);

    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        parents_[a].clear();

        dTree_[a].init(is, a);
    }
    
    /*初始化各数据结构空间*/
    dist_.reset(is); //

    classDist_.reset(is);

    pass_ = 1;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void cmkdb::train(const instance &inst) {
    if (pass_ == 1) {
        dist_.update(inst);
    } else {
        assert(pass_ == 2);

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            dTree_[a].update(inst, a, parents_[a]);
        }
        classDist_.update(inst);
    }
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void cmkdb::initialisePass() {
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void cmkdb::finalisePass() {
    if (pass_ == 1 && k_ != 0) {
        //printf("finalisePass\n");
        // calculate the mutual information from the xy distribution
        std::vector<float> mi;
        getMutualInformation(dist_.xyCounts, mi);

        if (verbosity >= 3) {
            printf("\nMutual information table\n");
            print(mi);
            printf("\n");
        }

        // calculate the conditional mutual information from the xxy distribution
        crosstab<float> cmi = crosstab<float>(noCatAtts_);
        getCondMutualInf(dist_, cmi);

        dist_.clear();

        //if (verbosity >= 3) {
            printf("\nConditional mutual information table\n");
            cmi.print();
        //}

        // sort the attributes on MI with the class
        std::vector<CategoricalAttribute> order;
        std::vector<CategoricalAttribute> order0; //left
        std::vector<CategoricalAttribute> order1; //right
        std::vector<CategoricalAttribute> orderatt;
        std::vector<CategoricalAttribute> add;
        std::vector<float> sumcmi;
        order0.resize(cmilimit_);
        order1.resize(cmilimit_);
        add.resize(cmilimit_);
        sumcmi.resize(cmilimit_);
        order0.clear();
        order1.clear();
        orderatt.clear();
        add.clear();
        sumcmi.clear();
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            order.push_back(a);
        }

        // assign the parents
        if (!order.empty()) {
            //miCmpClass cmp(&mi);

            //std::sort(order.begin(), order.end(), cmp);

            std::vector<bool> used;
            used.assign(noCatAtts_, false);

            printf("xi\txj\tcmi\n");
            for (CategoricalAttribute i = 0; i < cmilimit_; i++) {
                float maxcmi = 0;
                CategoricalAttribute pos1 = 0;
                CategoricalAttribute pos2 = 0;
                for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
                    for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
                        if (maxcmi < cmi[x1][x2]) {
                            maxcmi = cmi[x1][x2];
                            pos1 = x1;
                            pos2 = x2;
                        }
                    }
                }
                cmi[pos1][pos2] = 0.0;
                sumcmi[i] = maxcmi;
                printf("%d\t%d\t%f\n", pos1, pos2, maxcmi);
                if (i == 0 && mi[pos1] < mi[pos2]) {
                    CategoricalAttribute temp = pos1;
                    pos1 = pos2;
                    pos2 = temp;
                }
                
                order0.push_back(pos1);
                order1.push_back(pos2);
            }

            for (CategoricalAttribute i = 0; i < cmilimit_; i++) {
                //printf("%d\t%d\n", order0[i], order1[i]);
                //printf("cmi=%f\n",sumcmi[i]);
            }
            //printf("\nproper KDB assignment of parents First\n");
            parents_[order1[0]].push_back(order0[0]);
            used[order0[0]] = true;
            used[order1[0]] = true;
            add.assign(cmilimit_, 0);
            add[0] = 1;
            orderatt.push_back(order0[0]);
            orderatt.push_back(order1[0]);
            while (orderatt.size() <= noCatAtts_) {
                int ssize = 0;
                for (CategoricalAttribute i = 0; i < cmilimit_; i++) {
                    int size = orderatt.size();
                    ssize = size;
                    if (used[order0[i]] && used[order1[i]]) {
                        continue;
                    }


                    if (!used[order1[i]] && used[order0[i]]) {
                        for (CategoricalAttribute ai = 0; ai < size; ai++) {
                            if (order0[i] == orderatt[ai]) {
                                //printf("TF i=%d\tai=%d\n", i, ai);
                                parents_[order1[i]].push_back(orderatt[ai]);
                                add[i] = 1;
                                used[order1[i]] = true;
                                orderatt.push_back(order1[i]);
                                break;
                            }
                        }
                    }
                    if (!used[order0[i]] && used[order1[i]]) {
                        for (CategoricalAttribute ai = 0; ai < size; ai++) {
                            if (order1[i] == orderatt[ai]) {
                                //printf("FT i=%d\tai=%d\n", i, ai);
                                parents_[order0[i]].push_back(orderatt[ai]);
                                add[i] = 1;
                                used[order0[i]] = true;
                                orderatt.push_back(order0[i]);
                                break;
                            }
                        }
                    }
                }
                if (orderatt.size() == noCatAtts_)
                    break;
                if (ssize - orderatt.size() == 0)
                    break;
            }
            //printf("\nproper KDB assignment of parents Second\n");
            for (CategoricalAttribute i = 0; i < cmilimit_; i++) {
                //printf("%d\n",add[i]);
                CategoricalAttribute pos1 = 0;
                CategoricalAttribute pos2 = 0;
                for (CategoricalAttribute j = 0; j < noCatAtts_; j++) {
                    if (order0[i] == orderatt[j]) pos1 = j;
                    if (order1[i] == orderatt[j]) pos2 = j;
                }
                if (pos1 > pos2) {
                    CategoricalAttribute temp = pos1;
                    pos1 = pos2;
                    pos2 = temp;
                }
                if (add[i] == 0 && parents_[orderatt[pos2]].size() < k_) {
                    parents_[orderatt[pos2]].push_back(orderatt[pos1]);
                    add[i] = 1;
                }
            }
            
            for (CategoricalAttribute i = 0; i < cmilimit_; i++) {
                //printf("cmi=%f\n",sumcmi[i]);
                if (add[i]==1)
                    result += sumcmi[i];
            }
            printf("Sumcmi=%f\n",result);
            for (unsigned int i = 0; i < noCatAtts_; i++) {
                if (parents_[i].size() == 1) {
                    printf("parents[%d]=%d\n", i, parents_[i][0]);
                }
                if (parents_[i].size() > 1) {
                    printf("parents[%d][0]=%d\tparents[%d][1]=%d\n", i, parents_[i][0], i, parents_[i][1]);
                }
            }
        }
    }

    ++pass_;
}

/// true iff no more passes are required. updated by finalisePass()

bool cmkdb::trainingIsFinished() {
    return pass_ > 2;
}

void cmkdb::classify(const instance& inst, std::vector<double> &posteriorDist) {
    //printf("classify\n");
    // calculate the class probabilities in parallel
    // P(y)
    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
    }

    // P(x_i | x_p1, .. x_pk, y)
    for (CategoricalAttribute x = 0; x < noCatAtts_; x++) {
        //没用到加权平均啊
        dTree_[x].updateClassDistribution(posteriorDist, x, inst);
    }

    // normalise the results
    normalise(posteriorDist);
}


