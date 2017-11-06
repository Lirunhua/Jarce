
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "test_cmi.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

test_cmi::test_cmi() : pass_(1) {
}

test_cmi::test_cmi(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "test_cmi";
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

test_cmi::~test_cmi(void) {
}

void test_cmi::getCapabilities(capabilities &c) {
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

void test_cmi::reset(InstanceStream &is) {
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1

    // initialise distributions
    parents_.resize(noCatAtts);


    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        parents_[a].clear();
    }

    /*初始化各数据结构空间*/
    dist_1.reset(is); //
    classDist_.reset(is);
    trainingIsFinished_ = false;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void test_cmi::train(const instance &inst) {
    dist_1.update(inst);
    classDist_.update(inst);

}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void test_cmi::initialisePass() {
    assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void test_cmi::finalisePass() {
    assert(trainingIsFinished_ == false);
    //  printf("finalisePass\n");
    // calculate the mutual information from the xy distribution
    std::vector<float> mi;
    getMutualInformation(dist_1.xxyCounts.xyCounts, mi);    //互信息I(Xi,Xj))
    //print(mi);
    crosstab<float> mi_xxy = crosstab<float>(noCatAtts_); //联合互信息I(Xi,Xj;Y)
    getUnionMI(dist_1.xxyCounts, mi_xxy);
    double mi_sum = 0.0;
    for (CategoricalAttribute xi = 0; xi < noCatAtts_; xi++) {
        for (CategoricalAttribute xj = 0; xj < noCatAtts_; xj++) {
            mi_sum += mi_xxy[xi][xj];
        }
    }
    crosstab<float> mi_xxy_percent = crosstab<float>(noCatAtts_); //联合互信息百分比率I(Xi,Xj;Y)/H(Xi,Xj)
    for (CategoricalAttribute xi = 0; xi < noCatAtts_; xi++) {
        for (CategoricalAttribute xj = 0; xj < noCatAtts_; xj++) {
            mi_xxy_percent[xi][xj] = mi_xxy[xi][xj] / mi_sum;
        }
    }

    //crosstab<float> cmi = crosstab<float>(noCatAtts_);
    crosstab<float> cmi_xxy = crosstab<float>(noCatAtts_); //条件互信息I(Xi,Xj|Y)
    crosstab<float> cmi_xyx = crosstab<float>(noCatAtts_); //非类条件互信息I(Xi,Y|Xj)
    crosstab<float> cmi_xyx_ratio = crosstab<float>(noCatAtts_);
    crosstab<float> cmi_xxy_ratio = crosstab<float>(noCatAtts_);
    //crosstab<float> cmi_xxy_Ixx = crosstab<float>(noCatAtts_);
    //getCondMutualInf(dist_1.xxyCounts, cmi);
    getCMIxyx(dist_1.xxyCounts, cmi_xyx, cmi_xyx_ratio);
    getCMIxxy(dist_1.xxyCounts, cmi_xxy, cmi_xxy_ratio);
    
    double cmi_xyx_sum = 0.0;
    for (CategoricalAttribute xi = 0; xi < noCatAtts_; xi++) {
        for (CategoricalAttribute xj = 0; xj < noCatAtts_; xj++) {
           cmi_xyx_sum += cmi_xyx[xi][xj];
        }
    }
    crosstab<float> cmi_xyx_percent = crosstab<float>(noCatAtts_); //联合互信息百分比率
    for (CategoricalAttribute xi = 0; xi < noCatAtts_; xi++) {
        for (CategoricalAttribute xj = 0; xj < noCatAtts_; xj++) {
            cmi_xyx_percent[xi][xj] = cmi_xyx[xi][xj] / cmi_xyx_sum;
        }
    }
    double cmi_xxy_sum = 0.0;
    for (CategoricalAttribute xi = 0; xi < noCatAtts_; xi++) {
        for (CategoricalAttribute xj = 0; xj < noCatAtts_; xj++) {
           cmi_xxy_sum += cmi_xxy[xi][xj];
        }
    }
    crosstab<float> cmi_xxy_percent = crosstab<float>(noCatAtts_); //联合互信息百分比率
    for (CategoricalAttribute xi = 0; xi < noCatAtts_; xi++) {
        for (CategoricalAttribute xj = 0; xj < noCatAtts_; xj++) {
            cmi_xxy_percent[xi][xj] = cmi_xxy[xi][xj] / cmi_xxy_sum;
        }
    }
//    cmi_xxy_percent.print();
//    mi_xxy_percent.print();
//    cmi_xyx_percent.print();
    

    double maxmi = 0.0;
    double mi_temp = 0.0;
    CategoricalAttribute root1 = 0xFFFFFFFFUL;
    CategoricalAttribute root2 = 0xFFFFFFFFUL;
    for (CategoricalAttribute xi = 1; xi < noCatAtts_; xi++) {
        for (CategoricalAttribute xj = 0; xj < xi; xj++) {
            mi_temp = mi_xxy_percent[xi][xj] + cmi_xxy_percent[xi][xj];
            //printf("%d\t%d\t%lf\n", xi, xj, mi_temp);
            if (maxmi < mi_temp) {
                maxmi = mi_temp;
                root1 = xi;
                root2 = xj;
            }
        }
    }
    //printf("root12: %d\t%d\n", root1, root2);


    std::vector<CategoricalAttribute> used_nodes;
    used_nodes.push_back(root1);
    used_nodes.push_back(root2);
    //used_nodes.push_back(root3);
    parents_[root1].push_back(root2);
    std::vector<bool> waiting_nodes;
    waiting_nodes.assign(noCatAtts_, true);
    //waiting_nodes[root3] = false;
    waiting_nodes[root1] = false;
    waiting_nodes[root2] = false;
    
    while (used_nodes.size() != noCatAtts_) {
        //printf("%d\n", used_nodes.size());
        double maxcmi = 0.0;
        double temp = 0.0;
        CategoricalAttribute choose_child = 0xFFFFFFFFUL;
        CategoricalAttribute choose_father = 0xFFFFFFFFUL;
        for (CategoricalAttribute xi = 0; xi < used_nodes.size(); xi++) { //待选父节点
            for (CategoricalAttribute xj = 0; xj < noCatAtts_; xj++) {
                if (waiting_nodes[xj]) { //如果是待选子节点
                    temp = cmi_xxy_percent[used_nodes[xi]][xj] + cmi_xyx_percent[xj][used_nodes[xi]];
                    if (maxcmi < temp) {
                        maxcmi = temp;
                        choose_child = xj;
                        choose_father = used_nodes[xi];
                    }
                }
            }
        }
        //printf("choose %d\t%d\n", choose_child, choose_father);
        if (choose_child != 0xFFFFFFFFUL && choose_father != 0xFFFFFFFFUL) {
            parents_[choose_child].push_back(choose_father);
            used_nodes.push_back(choose_child);
            waiting_nodes[choose_child] = false;
        }
    }
    printf("parents_:\n");
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        if (parents_[i].size() == 0) {
            printf("parents_[%d][0]\tY\n", i);
        }
        if (parents_[i].size() == 1) {
            printf("parents_[%d][0]=%d\n", i, parents_[i][0]);
        }
        if (parents_[i].size() == 2) {
            printf("parents_[%d][0]=%d\tparents_[%d][1]=%d\n", i, parents_[i][0], i, parents_[i][1]);
        }
    }

    //order.clear();
    used_nodes.clear();

    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool test_cmi::trainingIsFinished() {
    return trainingIsFinished_;
}

void test_cmi::classify(const instance& inst, std::vector<double> &posteriorDist) {
    // printf("classify\n");


    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
    }
    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents_[x1].size() == 0) {
                // printf("PARent=0  \n");
                posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
            } else if (parents_[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents_[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]), y);
                }
            }

        }
    }

    // normalise the results
    normalise(posteriorDist);
}
