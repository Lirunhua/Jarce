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

#include "decisiontreekdb.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

decisiontreekdb::decisiontreekdb() : pass_(1) {
}

decisiontreekdb::decisiontreekdb(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "decisiontreekdb";

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

decisiontreekdb::~decisiontreekdb(void) {
}

void decisiontreekdb::getCapabilities(capabilities &c) {
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

void decisiontreekdb::reset(InstanceStream &is) {
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();
    numofAttvalue = 0;
    root = 0;
    root_loc = 0;
    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1

    // initialise distributions

    /*初始化各数据结构空间*/
    dist_.reset(is); //
    dist_1.reset(is);
    //dist_2.reset(is);
    classDist_.reset(is);
    trainingIsFinished_ = false;
    //pass_ = 1;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void decisiontreekdb::train(const instance &inst) {
    dist_.update(inst);
    dist_1.update(inst);
    //dist_2.update(inst);
    classDist_.update(inst);
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void decisiontreekdb::initialisePass() {
    assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void decisiontreekdb::finalisePass() {

    assert(trainingIsFinished_ == false);
    // calculate the mutual information from the xy distribution

    std::vector<float> mi;
    getMutualInformation(dist_.xxyCounts.xyCounts, mi);

    if (verbosity >= 3) {
        printf("\nMutual information table\n");
        print(mi);
    }

    // sort the attributes on MI with the class
    std::vector<CategoricalAttribute> order;

    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        order.push_back(a);
    }

    // assign the parents
    if (!order.empty()) {
        miCmpClass cmp(&mi);

        std::sort(order.begin(), order.end(), cmp);

        numofAttvalue = dist_.getNoValues(order[0]);
        root = order[0];
        //printf("\n");
        //printf("order[0]=%d\tnumofAttvalue=%d\n", order[0], numofAttvalue);
        //printf("order[0]->getCatAttName=%s\n", instanceStream_->getCatAttName(order[0]));
        parents_.resize(numofAttvalue);
        for (CategoricalAttribute a = 0; a < numofAttvalue; a++) {
            parents_[a].resize(noCatAtts_);
        }
        for (CategoricalAttribute a = 0; a < numofAttvalue; a++) {
            for (CategoricalAttribute b = 0; b < noCatAtts_; b++) {
                parents_[a][b].clear();
            }
        }

        for (unsigned int attvalno = 0; attvalno < numofAttvalue; attvalno++) {

            std::vector<CategoricalAttribute> suborder;

            for (CategoricalAttribute a = 0; a < noCatAtts_ - 1; a++) {
                suborder.push_back(a);
            }

            std::vector<float> xmi;
            getXMI(dist_.xxyCounts, xmi, order[0], attvalno);

            crosstab<float> xcmi = crosstab<float>(noCatAtts_);
            getXCMI(dist_, xcmi, order[0], attvalno);

            if (!suborder.empty()) {

                miCmpClass cmp(&xmi);

                std::sort(suborder.begin(), suborder.end(), cmp);
                //printf("order[0]=%d\tsuborder[0]=%d\n", order[0], suborder[0]);
                parents_[attvalno][suborder[0]].push_back(order[0]);
                // proper KDB assignment of parents
                for (std::vector<CategoricalAttribute>::const_iterator it = suborder.begin() + 1; it != suborder.end(); it++) {
                    parents_[attvalno][*it].push_back(suborder[0]);
                    for (std::vector<CategoricalAttribute>::const_iterator it2 = suborder.begin() + 1; it2 != it; it2++) {
                        // make parents into the top k attributes on mi that precede *it in order
                        if (parents_[attvalno][*it].size() < k_) {
                            // create space for another parent
                            // set it initially to the new parent.
                            // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
                            parents_[attvalno][*it].push_back(*it2);
                        }
                        for (unsigned int i = 0; i < parents_[attvalno][*it].size(); i++) {
                            if (xcmi[*it2][*it] > xcmi[parents_[attvalno][*it][i]][*it]) {
                                // move lower value parents down in order
                                for (unsigned int j = parents_[attvalno][*it].size() - 1; j > i; j--) {
                                    parents_[attvalno][*it][j] = parents_[attvalno][*it][j - 1];
                                }
                                // insert the new att
                                parents_[attvalno][*it][i] = *it2;
                                break;
                            }
                        }
                    }


                }
            }
            /*printf("\n");
            for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
                if (parents_[attvalno][i].size() == 1) {
                    printf("parents[%d][%d][0]=%d\n", attvalno, i, parents_[attvalno][i][0]);
                }
                if (parents_[attvalno][i].size() > 1) {
                    for (unsigned int w = 0; w < parents_[attvalno][i].size(); w++) {
                        printf("parents[%d][%d][%d]=%d\t", attvalno, i, w, parents_[attvalno][i][w]);
                    }
                    printf("\n");
                }
            }*/

        }
    }
    dist_.clear();

    trainingIsFinished_ = true;
    //++pass_;
}

/// true iff no more passes are required. updated by finalisePass()

bool decisiontreekdb::trainingIsFinished() {
    return trainingIsFinished_;
}

void decisiontreekdb::classify(const instance& inst, std::vector<double> &posteriorDist) {

    //printf("inst.getCatVal(%d)=%d\n", root, inst.getCatVal(root));
    CategoricalAttribute attvalno = inst.getCatVal(root);


    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents_[attvalno][x1].size() == 0) {

                posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate

            } else if (parents_[attvalno][x1].size() == 1) {
                const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_[attvalno][x1][0], inst.getCatVal(parents_[attvalno][x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[attvalno][x1][0], inst.getCatVal(parents_[attvalno][x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents_[attvalno][x1].size() == 2) {
                const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_[attvalno][x1][0], inst.getCatVal(parents_[attvalno][x1][0]), parents_[attvalno][x1][1], inst.getCatVal(parents_[attvalno][x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_[attvalno][x1][0], inst.getCatVal(parents_[attvalno][x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[attvalno][x1][0], inst.getCatVal(parents_[attvalno][x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_[attvalno][x1][0], inst.getCatVal(parents_[attvalno][x1][0]), parents_[attvalno][x1][1], inst.getCatVal(parents_[attvalno][x1][1]), y);
                }
            }
        }
    }



 
    //局部！！！！！！！！！
    std::vector<float> mi_loc;
    getMutualInformationloc(dist_1.xxyCounts.xyCounts, mi_loc, inst);

    if (verbosity >= 3) {
        printf("\nMutual information table\n");
        print(mi_loc);
    }

    // sort the attributes on MI with the class
    std::vector<CategoricalAttribute> order_loc;

    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        order_loc.push_back(a);
    }
 
    // assign the parents
    if (!order_loc.empty()) {
        miCmpClass cmp(&mi_loc);

        std::sort(order_loc.begin(), order_loc.end(), cmp);
 
        root_loc = order_loc[0];
        CategoricalAttribute attvalno_loc = inst.getCatVal(order_loc[0]);
 
        //printf("\n");
        //printf("order[0]=%d\tnumofAttvalue=%d\n", order[0], numofAttvalue);
        //printf("order[0]->getCatAttName=%s\n", instanceStream_->getCatAttName(order[0]));
        parents_loc.resize(noCatAtts_);
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            parents_loc[a].clear();
        }
 

        std::vector<CategoricalAttribute> suborder_loc;

        for (CategoricalAttribute a = 0; a < noCatAtts_ - 1; a++) {
            suborder_loc.push_back(a);
        }
 
        std::vector<float> xmi_loc;
        getXMI_loc(dist_1.xxyCounts, xmi_loc, order_loc[0], attvalno_loc, inst);
 
        crosstab<float> xcmi_loc = crosstab<float>(noCatAtts_);
        getXCMI_loc(dist_1, xcmi_loc, order_loc[0], attvalno_loc, inst);
       
        if (!suborder_loc.empty()) {

            miCmpClass cmp(&xmi_loc);
            
            std::sort(suborder_loc.begin(), suborder_loc.end(), cmp);
            //printf("order[0]=%d\tsuborder[0]=%d\n", order[0], suborder[0]);
            parents_loc[suborder_loc[0]].push_back(order_loc[0]);
            // proper KDB assignment of parents
            for (std::vector<CategoricalAttribute>::const_iterator it = suborder_loc.begin() + 1; it != suborder_loc.end(); it++) {
                parents_loc[*it].push_back(suborder_loc[0]);
                for (std::vector<CategoricalAttribute>::const_iterator it2 = suborder_loc.begin() + 1; it2 != it; it2++) {
                    // make parents into the top k attributes on mi that precede *it in order
                    if (parents_loc[*it].size() < k_) {
                        // create space for another parent
                        // set it initially to the new parent.
                        // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
                        parents_loc[*it].push_back(*it2);
                    }
                    for (unsigned int i = 0; i < parents_loc[*it].size(); i++) {
                        if (xcmi_loc[*it2][*it] > xcmi_loc[parents_loc[*it][i]][*it]) {
                            // move lower value parents down in order
                            for (unsigned int j = parents_loc[*it].size() - 1; j > i; j--) {
                                parents_loc[*it][j] = parents_loc[*it][j - 1];
                            }
                            // insert the new att
                            parents_loc[*it][i] = *it2;
                            break;
                        }
                    }
                }
            }
        }
    }


    std::vector<double> posteriorDist1;
    posteriorDist1.assign(noClasses_, 0);



    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist1[y] = dist_1.xxyCounts.xyCounts.p(y)* (std::numeric_limits<double>::max() / 2.0);

    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents_loc[x1].size() == 0) {
                // printf("PARent=0  \n");

                posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate


            } else if (parents_loc[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_loc[x1][0], inst.getCatVal(parents_loc[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist1[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_loc[x1][0], inst.getCatVal(parents_loc[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents_loc[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_loc[x1][0], inst.getCatVal(parents_loc[x1][0]), parents_loc[x1][1], inst.getCatVal(parents_loc[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_loc[x1][0], inst.getCatVal(parents_loc[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist1[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_loc[x1][0], inst.getCatVal(parents_loc[x1][0]), y);
                    }
                } else {
                    posteriorDist1[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_loc[x1][0], inst.getCatVal(parents_loc[x1][0]), parents_loc[x1][1], inst.getCatVal(parents_loc[x1][1]), y);
                }

            }
        }
    }

    for (int classno = 0; classno < noClasses_; classno++) {

        posteriorDist[classno] += posteriorDist1[classno];
        posteriorDist[classno] = posteriorDist[classno] / 2.0;

    }

    normalise(posteriorDist);

}


