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

#include "entropykdb.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

entropykdb::entropykdb() : pass_(1) {
}

entropykdb::entropykdb(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "entropykdb";

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

entropykdb::~entropykdb(void) {
}

void entropykdb::getCapabilities(capabilities &c) {
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

void entropykdb::reset(InstanceStream &is) {
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

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
void entropykdb::train(const instance &inst) {
    if (pass_ == 1) {
        dist_.update(inst);
        //hDist_.update(inst);
    } else {
        assert(pass_ == 2);

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            dTree_[a].update(inst, a, parents_[a]);
        }
        classDist_.update(inst);
    }
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void entropykdb::initialisePass() {
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void entropykdb::finalisePass() {
    if (pass_ == 1 && k_ != 0) {

        // calculate the mutual information from the xy distribution
        std::vector<float> entropy;
        //getMutualInformation(dist_.xyCounts, mi);

        std::vector<InstanceCount> InsCount;
        //double h = 0;
        //std::vector<std::vector<CategoricalAttribute> > attno_;
        //attno_.resize(noClasses_);
        int c = 0;

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            double result = 0;
            for (CategoricalAttribute attval = 0; attval < dist_.getNoValues(a); attval++) {
                for (CategoricalAttribute label = 0; label < noClasses_; label++) {
                    c = dist_.xyCounts.getCount(a, attval, label);
                    //printf("%d\t%d\t%d\t%d\n",a,attval,label,c);
                    InsCount.push_back(c);
                }
                result += getEntropy(InsCount);
            }

            //printf("%f\n",result);
            entropy.push_back(result);
        }

        //printf("entropy:");
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            //printf("%g\t", entropy[a]);

        }
        //printf("\n");

        if (verbosity >= 3) {
            printf("\nMutual information table\n");
            //print(mi);
        }

        // calculate the conditional mutual information from the xxy distribution
        crosstab<float> cmi = crosstab<float>(noCatAtts_);
        getCondMutualInf(dist_, cmi);

        dist_.clear();

        if (verbosity >= 3) {
            printf("\nConditional mutual information table\n");
            cmi.print();
        }

        // sort the attributes on MI with the class
        std::vector<CategoricalAttribute> order;

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            order.push_back(a);
        }

        // assign the parents
        if (!order.empty()) {
            miCmpClass cmp(&entropy);

            std::sort(order.begin(), order.end(), cmp);
            std::reverse(order.begin(), order.end());
            //printf("order:\t");
            for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                //printf("%d\t", order[a]);

            }
            //printf("\n");
            if (verbosity >= 2) {
                printf("\n%s parents:\n", instanceStream_->getCatAttName(order[0]));
            }

            // proper entropykdb assignment of parents
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

            for (unsigned int i = 0; i < noCatAtts_; i++) {
                if (parents_[i].size() == 1) {
                    //printf("parents[%d]=%d\n", i, parents_[i][0]);
                    result += cmi[i][parents_[i][0]];
                }
                if (parents_[i].size() > 1) {
                    //printf("parents[%d][0]=%d\tparents[%d][1]=%d\n", i, parents_[i][0], i, parents_[i][1]);
                    result += cmi[i][parents_[i][0]];
                    result += cmi[i][parents_[i][1]];
                }
            }
            //printf("Sum_cmi=%f\n", result);
        }
    }

    ++pass_;
}

/// true iff no more passes are required. updated by finalisePass()

bool entropykdb::trainingIsFinished() {
    return pass_ > 2;
}

void entropykdb::classify(const instance& inst, std::vector<double> &posteriorDist) {
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