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

#include "localkdb.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

localkdb::localkdb() : pass_(1) {
}

localkdb::localkdb(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "localkdb";
    union_kdb_localkdb = false;
    // defaults
    k_ = 1;

    // get arguments
    while (argv != end) {
        if (*argv[0] != '-') {
            break;
        } else if (argv[0][1] == 'k') {
            getUIntFromStr(argv[0] + 2, k_, "k");
        } else if (streq(argv[0] + 1, "un")) {
            union_kdb_localkdb = true;
        } else {
            break;
        }

        name_ += argv[0];

        ++argv;
    }
}

localkdb::~localkdb(void) {
}

void localkdb::getCapabilities(capabilities &c) {
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

void localkdb::reset(InstanceStream &is) {
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1

    // initialise distributions
    dTree_.resize(noCatAtts);
    //dTree_1.resize(noCatAtts);
    parents_.resize(noCatAtts);
    //  parents_1.resize(noCatAtts);



    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        parents_[a].clear();
        //  parents_1[a].clear();
        dTree_[a].init(is, a);
        //  dTree_1[a].init(is, a);
    }

    /*初始化各数据结构空间*/
    dist_1.reset(is); //
    classDist_.reset(is);
    //  classDist_1.reset(is);
    trainingIsFinished_ = false;
    //pass_ = 1;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void localkdb::train(const instance &inst) {
    dist_1.update(inst);
    classDist_.update(inst);

}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void localkdb::initialisePass() {
    assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void localkdb::finalisePass() {
    assert(trainingIsFinished_ == false);
    //  printf("finalisePass\n");
    //if (pass_ == 1 && k_ != 0) {
    // calculate the mutual information from the xy distribution
    std::vector<float> mi;
    getMutualInformation(dist_1.xxyCounts.xyCounts, mi);

    // calculate the conditional mutual information from the xxy distribution
    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    getCondMutualInf(dist_1.xxyCounts, cmi);

    // sort the attributes on MI with the class
    std::vector<CategoricalAttribute> order;

    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        order.push_back(a);
    }

    // assign the parents
    if (!order.empty()) {
        miCmpClass cmp(&mi);

        std::sort(order.begin(), order.end(), cmp);

        // proper KDB assignment of parents
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

            if (verbosity >= 2) {
                printf("%s parents: ", instanceStream_->getCatAttName(*it));
                for (unsigned int i = 0; i < parents_[*it].size(); i++) {
                    printf("%s ", instanceStream_->getCatAttName(parents_[*it][i]));
                }
                putchar('\n');
            }
        }
    }
    order.clear();

    //}
    trainingIsFinished_ = true;
    //++pass_;
}

/// true iff no more passes are required. updated by finalisePass()

bool localkdb::trainingIsFinished() {
    return trainingIsFinished_;
    //return pass_ > 2;
}

void localkdb::classify(const instance& inst, std::vector<double> &posteriorDist) {
    // printf("classify\n");
    //局部kdb

    parents_1.resize(noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_1[a].clear();
    }

    std::vector<float> mi_loc;
    getMutualInformationloc(dist_1.xxyCounts.xyCounts, mi_loc, inst);

    // calculate the conditional mutual information from the xxy distribution
    crosstab<float> cmi_loc = crosstab<float>(noCatAtts_);
    getCondMutualInfloc(dist_1.xxyCounts, cmi_loc, inst);

    // sort the attributes on MI with the class
    std::vector<CategoricalAttribute> order1;

    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        order1.push_back(a);
    }

    // assign the parents
    if (!order1.empty()) {
        miCmpClass cmp(&mi_loc);

        std::sort(order1.begin(), order1.end(), cmp);

        // proper KDB assignment of parents
        for (std::vector<CategoricalAttribute>::const_iterator it = order1.begin() + 1; it != order1.end(); it++) {
            parents_1[*it].push_back(order1[0]);
            for (std::vector<CategoricalAttribute>::const_iterator it2 = order1.begin() + 1; it2 != it; it2++) {
                // make parents into the top k attributes on mi that precede *it in order1
                if (parents_1[*it].size() < k_) {
                    // create space for another parent
                    // set it initially to the new parent.
                    // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
                    parents_1[*it].push_back(*it2);
                }
                for (unsigned int i = 0; i < parents_1[*it].size(); i++) {
                    if (cmi_loc[*it2][*it] > cmi_loc[parents_1[*it][i]][*it]) {
                        // move lower value parents down in order1
                        for (unsigned int j = parents_1[*it].size() - 1; j > i; j--) {
                            parents_1[*it][j] = parents_1[*it][j - 1];
                        }
                        // insert the new att
                        parents_1[*it][i] = *it2;
                        break;
                    }
                }
            }
        }
    }

    order1.clear();


    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = dist_1.xxyCounts.xyCounts.p(y)* (std::numeric_limits<double>::max() / 2.0);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents_1[x1].size() == 0) {
                // printf("PARent=0  \n");
                posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
            } else if (parents_1[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents_1[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), parents_1[x1][1], inst.getCatVal(parents_1[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), parents_1[x1][1], inst.getCatVal(parents_1[x1][1]), y);
                }
            }
        }
    }
    //printf("parents_1 work over!\n");

    // normalise the results
    normalise(posteriorDist);


    if (union_kdb_localkdb == true) {
        //全局parents_的联合概率
        std::vector<double> posteriorDist1;
        posteriorDist1.assign(noClasses_, 0);

        for (CatValue y = 0; y < noClasses_; y++) {
            posteriorDist1[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
        }
        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
            for (CatValue y = 0; y < noClasses_; y++) {
                if (parents_[x1].size() == 0) {
                    // printf("PARent=0  \n");
                    posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
                } else if (parents_[x1].size() == 1) {
                    //  printf("PARent=1  \n");
                    const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                    if (totalCount1 == 0) {
                        posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist1[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                    }
                } else if (parents_[x1].size() == 2) {
                    // printf("PARent=2  \n");
                    const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]));
                    if (totalCount1 == 0) {
                        const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                        if (totalCount2 == 0) {
                            posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                        } else {
                            posteriorDist1[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y);
                        }
                    } else {
                        posteriorDist1[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]), y);
                    }
                }
            }
        }
        normalise(posteriorDist1);

        //联合概率结合
        for (int classno = 0; classno < noClasses_; classno++) {
            posteriorDist[classno] += posteriorDist1[classno];
            posteriorDist[classno] = posteriorDist[classno] / 2;
        }
    }

}



