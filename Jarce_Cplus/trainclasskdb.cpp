#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "trainclasskdb.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

trainclasskdb::trainclasskdb() : pass_(1) {
}

trainclasskdb::trainclasskdb(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "trainclasskdb";

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

trainclasskdb::~trainclasskdb(void) {
}

void trainclasskdb::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

// creates a comparator for two attributes based on their relative mutual information with the class

class miCmpClass1 {
public:

    miCmpClass1(std::vector<double> *m) {
        mi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
        return (*mi)[a] > (*mi)[b];
    }

private:
    std::vector<double> *mi;
};

void trainclasskdb::reset(InstanceStream &is) {
    //printf("reset\n");
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    //const unsigned int noCatAtts = 5;
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();
    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1
    arc = 0;

    /*初始化各数据结构空间*/
    dist_.reset(is); //
    classDist_.reset(is);
    trainingIsFinished_ = false;

    parents_.resize(noClasses_);
    parents_final.resize(noClasses_);
    for (CategoricalAttribute a = 0; a < noClasses_; a++) {
        parents_[a].resize(noCatAtts);
        parents_final[a].resize(noCatAtts);
    }
    for (CategoricalAttribute a = 0; a < noClasses_; a++) {
        for (CategoricalAttribute b = 0; b < noCatAtts; b++) {
            parents_[a][b].clear();
            parents_final[a][b].clear();
        }
    }
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void trainclasskdb::train(const instance &inst) {
    dist_.update(inst);
    classDist_.update(inst);
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void trainclasskdb::initialisePass() {
    assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void trainclasskdb::finalisePass() {
    //printf("finalisePass\n");
    assert(trainingIsFinished_ == false);
    int maxitem = 0; //解决栈溢出
    if (noCatAtts_ > noClasses_)
        maxitem = noCatAtts_;
    else
        maxitem = noClasses_;
    // calculate the mutual information from the xy distribution
    crosstab<double> mi = crosstab<double>(maxitem);

    getMutualInformationTC(dist_.xxyCounts.xyCounts, mi); //一维->二维
    //    mi.print();
    crosstab3D<double> cmi = crosstab3D<double>(maxitem);

    getCondMutualInfTC(dist_.xxyCounts, cmi); //二维->三维

    // sort the attributes on MI with the class
    // assign the parents
    for (CategoricalAttribute c = 0; c < noClasses_; c++) {
        //printf("%d\n", c);
        std::vector<double> mi_;
        mi_.assign(noCatAtts_, 0.0);

        std::vector<CategoricalAttribute> order;
        order.clear();
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            order.push_back(a);
        }

        miCmpClass1 cmp1(&mi_);
        std::sort(order.begin(), order.end(), cmp1);

        if (!order.empty()) {
            // proper KDB assignment of parents
            for (std::vector<CategoricalAttribute>::const_iterator it = order.begin() + 1; it != order.end(); it++) {
                parents_[c][*it].push_back(order[0]);
                for (std::vector<CategoricalAttribute>::const_iterator it2 = order.begin() + 1; it2 != it; it2++) {
                    // make parents into the top k attributes on mi that precede *it in order
                    if (parents_[c][*it].size() < k_) {
                        // create space for another parent
                        // set it initially to the new parent.
                        // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
                        parents_[c][*it].push_back(*it2);
                    }
                    for (unsigned int i = 0; i < parents_[c][*it].size(); i++) {
                        if (cmi[c][*it2][*it] > cmi[c][parents_[c][*it][i]][*it]) {
                            // move lower value parents down in order
                            for (unsigned int j = parents_[c][*it].size() - 1; j > i; j--) {
                                parents_[c][*it][j] = parents_[c][*it][j - 1];
                            }
                            // insert the new att
                            parents_[c][*it][i] = *it2;
                            break;
                        }
                    }
                }

            }

        }

        //        printf("Class = %d\n", c);
        //        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        //            if (parents_[c][x1].size() == 0) {
        //                printf("%d has no parents\n", x1);
        //            }
        //            if (parents_[c][x1].size() == 1) {
        //                printf("%d->%d\n", parents_[c][x1][0], x1);
        //            }
        //            if (parents_[c][x1].size() == 2) {
        //                printf("%d->%d,%d->%d\n", parents_[c][x1][0], x1, parents_[c][x1][1], x1);
        //            }
        //        }
    }

    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool trainclasskdb::trainingIsFinished() {
    return trainingIsFinished_;
}

void trainclasskdb::classify(const instance& inst, std::vector<double> &posteriorDist) {
    //    printf("classify\n");
    // calculate the class probabilities in parallel
    // P(y)
    for (CategoricalAttribute c = 0; c < noClasses_; c++) {
        parents_final[c].resize(noCatAtts_);
    }
    for (CategoricalAttribute a = 0; a < noClasses_; a++) {
        for (CategoricalAttribute b = 0; b < noCatAtts_; b++) {
            parents_final[a][b].clear();
        }
    }

    //    parents_1.resize(noCatAtts_);
    //    parents_2.resize(noCatAtts_);
    //
    //    for (CatValue c = 0; c < noClasses_; c++) {
    //
    //        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
    //            parents_1[a].clear();
    //        }
    //        
    //        //从三维中取出一层
    //        for (unsigned int a = 0; a < noCatAtts_; a++) {
    //            for (unsigned int b = 0; b < parents_[c][a].size(); b++) {
    //                parents_1[a].push_back(parents_[c][a][b]);
    //            }
    //        }
    //        arc = 0;
    //        for (unsigned int i = 0; i < noCatAtts_; i++) {
    //            if (parents_1[i].size() == 1) {
    //                arc++;
    //            }
    //            if (parents_1[i].size() > 1) {
    //                arc += k_;
    //            }
    //        }
    //        
    //        double H_standard = 0.0;
    //        H_standard = H_standard_loc_k2(classDist_, dist_, parents_1, inst);
    //        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
    //            parents_2[a].clear();
    //        }
    //        H_recursion_dp_k2(classDist_, dist_, H_standard, arc, order_[c], parents_1, parents_2, inst);
    //        //放入该层到三维
    //        for (unsigned int a = 0; a < noCatAtts_; a++) {
    //            for (unsigned int b = 0; b < parents_2[a].size(); b++) {
    //                parents_final[c][a].push_back(parents_2[a][b]);
    //            }
    //        }
    //        //        printf("parents_2:\n");
    //        //        for (unsigned int i = 0; i < noCatAtts_; i++) {
    //        //            if (parents_2[i].size() == 0) {
    //        //                printf("parents_2[%d][0]\ty=%d\n", i,c);
    //        //            }
    //        //            if (parents_2[i].size() == 1) {
    //        //                printf("parents_2[%d][0]=%d\n", i, parents_2[i][0]);
    //        //            }
    //        //            if (parents_2[i].size() == 2) {
    //        //                printf("parents_2[%d][0]=%d\tparents_2[%d][1]=%d\n", i, parents_2[i][0], i, parents_2[i][1]);
    //        //            }
    //        //            if (parents_2[i].size() == 3) {
    //        //                printf("parents_2[%d][0]=%d\tparents_2[%d][1]=%d\tparents_2[%d][2]=%d\n", i, parents_2[i][0], i, parents_2[i][1], i, parents_2[i][2]);
    //        //            }
    //        //        }
    //    }
    for (CatValue c = 0; c < noClasses_; c++) {
        for (unsigned int a = 0; a < noCatAtts_; a++) {
            for (unsigned int b = 0; b < parents_[c][a].size(); b++) {
                parents_final[c][a].push_back(parents_[c][a][b]);
            }
        }
    }
    //    for (CatValue c = 0; c < noClasses_; c++) {
    //        printf("Class = %d\n", c);
    //        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
    //            if (parents_final[c][x1].size() == 0) {
    //                printf("%d has no parents\n", x1);
    //            }
    //            if (parents_final[c][x1].size() == 1) {
    //                printf("%d->%d\n", parents_final[c][x1][0], x1);
    //            }
    //            if (parents_final[c][x1].size() == 2) {
    //                printf("%d->%d,%d->%d\n", parents_final[c][x1][0], x1, parents_final[c][x1][1], x1);
    //            }
    //
    //        }
    //    }
    //    printf("---------------------------------------\n");
    //    联合概率
    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
    }
    for (CatValue c = 0; c < noClasses_; c++) {
        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {

            if (parents_final[c][x1].size() == 0) {
                // printf("PARent=0  \n");
                posteriorDist[c] *= dist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), c); // p(a=v|Y=y) using M-estimate
            } else if (parents_final[c][x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist_.xxyCounts.xyCounts.getCount(parents_final[c][x1][0], inst.getCatVal(parents_final[c][x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[c] *= dist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), c);
                } else {
                    posteriorDist[c] *= dist_.xxyCounts.p(x1, inst.getCatVal(x1), parents_final[c][x1][0], inst.getCatVal(parents_final[c][x1][0]), c); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents_final[c][x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist_.xxyCounts.getCount(parents_final[c][x1][0], inst.getCatVal(parents_final[c][x1][0]), parents_final[c][x1][1], inst.getCatVal(parents_final[c][x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist_.xxyCounts.xyCounts.getCount(parents_final[c][x1][0], inst.getCatVal(parents_final[c][x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[c] *= dist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), c);
                    } else {
                        posteriorDist[c] *= dist_.xxyCounts.p(x1, inst.getCatVal(x1), parents_final[c][x1][0], inst.getCatVal(parents_final[c][x1][0]), c);
                    }
                } else {
                    posteriorDist[c] *= dist_.p(x1, inst.getCatVal(x1), parents_final[c][x1][0], inst.getCatVal(parents_final[c][x1][0]), parents_final[c][x1][1], inst.getCatVal(parents_final[c][x1][1]), c);
                }

            }
        }
    }

    normalise(posteriorDist);
    double max = -1;
    int pos = 0xFFFFFFFFUL;
    for (CatValue y = 0; y < noClasses_; y++) {
        if (max < posteriorDist[y]) {
            max = posteriorDist[y];
            pos = y;
        }
        printf("%lf\t", posteriorDist[y]);
    }
    printf("====>%d\n", pos);

    posteriorDist.assign(noClasses_, 0);
    crosstab<double> posteriorDist_2D = crosstab<double>(noClasses_);
    std::vector<double> posteriorDist_temp;
    //    posteriorDist_2D.print();
    //row--model col--P1P2P3...
    for (CatValue y_row = 0; y_row < noClasses_; y_row++) {
        posteriorDist_temp.assign(noClasses_, 0);
        for (CatValue y_col = 0; y_col < noClasses_; y_col++) {
            posteriorDist_2D[y_row][y_col] = classDist_.p(y_row) * (std::numeric_limits<double>::max() / 2.0);
        }
        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
            for (CatValue y = 0; y < noClasses_; y++) {
                if (parents_final[y_row][x1].size() == 0) {
                    posteriorDist_2D[y_row][y] *= dist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
                } else if (parents_final[y_row][x1].size() == 1) {
                    const InstanceCount totalCount1 = dist_.xxyCounts.xyCounts.getCount(parents_final[y_row][x1][0], inst.getCatVal(parents_final[y_row][x1][0]));
                    if (totalCount1 == 0) {
                        posteriorDist_2D[y_row][y] *= dist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist_2D[y_row][y] *= dist_.xxyCounts.p(x1, inst.getCatVal(x1), parents_final[y_row][x1][0], inst.getCatVal(parents_final[y_row][x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                    }
                } else if (parents_final[y_row][x1].size() == 2) {
                    const InstanceCount totalCount1 = dist_.xxyCounts.getCount(parents_final[y_row][x1][0], inst.getCatVal(parents_final[y_row][x1][0]), parents_final[y_row][x1][1], inst.getCatVal(parents_final[y_row][x1][1]));
                    if (totalCount1 == 0) {
                        const InstanceCount totalCount2 = dist_.xxyCounts.xyCounts.getCount(parents_final[y_row][x1][0], inst.getCatVal(parents_final[y_row][x1][0]));
                        if (totalCount2 == 0) {
                            posteriorDist_2D[y_row][y] *= dist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                        } else {
                            posteriorDist_2D[y_row][y] *= dist_.xxyCounts.p(x1, inst.getCatVal(x1), parents_final[y_row][x1][0], inst.getCatVal(parents_final[y_row][x1][0]), y);
                        }
                    } else {
                        posteriorDist_2D[y_row][y] *= dist_.p(x1, inst.getCatVal(x1), parents_final[y_row][x1][0], inst.getCatVal(parents_final[y_row][x1][0]), parents_final[y_row][x1][1], inst.getCatVal(parents_final[y_row][x1][1]), y);
                    }
                }
            }
        }
        for (CatValue y_col = 0; y_col < noClasses_; y_col++) {
            posteriorDist_temp[y_col] = posteriorDist_2D[y_row][y_col];
        }
        normalise(posteriorDist_temp);
        for (CatValue y_col = 0; y_col < noClasses_; y_col++) {
            posteriorDist_2D[y_row][y_col] = posteriorDist_temp[y_col];
        }
    }
    //    printf("******\n");
    posteriorDist_2D.print();
    for (CatValue y_col = 0; y_col < noClasses_; y_col++) {
        for (CatValue y_row = 0; y_row < noClasses_; y_row++) {
            posteriorDist[y_col] += posteriorDist_2D[y_row][y_col];
        }
        posteriorDist[y_col] = posteriorDist[y_col] / noClasses_;
    }
    max = -1;
    pos = 0xFFFFFFFFUL;
    for (CatValue y = 0; y < noClasses_; y++) {
        if (max < posteriorDist[y]) {
            max = posteriorDist[y];
            pos = y;
        }
        printf("%lf\t", posteriorDist[y]);
    }
    printf("====>%d\n", pos);
    printf("--------------------------------------------------------------\n");
}




