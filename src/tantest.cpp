#include "tantest.h"
#include "utils.h"
#include "correlationMeasures.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>

tantest::tantest() :
trainingIsFinished_(false) {
}

tantest::tantest(char* const *&, char* const *) :
xxyDist_(), trainingIsFinished_(false) {
    name_ = "tantest";
}

tantest::~tantest(void) {
}

void tantest::reset(InstanceStream &is) {
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

    xxyDist_.reset(is);
}

void tantest::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

class orderCmpClass {
public:

    orderCmpClass(std::vector<double> *m) {
        mi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
        return (*mi)[a] > (*mi)[b];
    }

private:
    std::vector<double> *mi;
};

void tantest::initialisePass() {
    assert(trainingIsFinished_ == false);
}

void tantest::train(const instance &inst) {
    xxyDist_.update(inst);
}

void tantest::finalisePass() {
    printf("*** finalisePass ***\n");
    assert(trainingIsFinished_ == false);

    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    getCondMutualInf(xxyDist_, cmi);

    // find the maximum spanning tree
    CategoricalAttribute firstAtt = 0;

    parents_[firstAtt] = NOPARENT;
    printf("begin->%d\n", firstAtt);
    float *maxWeight; //就是Prim里的lowcost 记录生成树到某个结点目前的最大权值
    CategoricalAttribute *bestSoFar; //记录目前给这个结点选的父结点
    CategoricalAttribute topCandidate = firstAtt; //这次要加入最大生成树的属性
    std::set<CategoricalAttribute> available; //尚未加入生成树的属性

    safeAlloc(maxWeight, noCatAtts_); //其实就是 maxWeight=new float[noCatAtts_]
    safeAlloc(bestSoFar, noCatAtts_);

    maxWeight[firstAtt] = -std::numeric_limits<float>::max();
    //Prim算法开始 完全图是稠密图，适合使用Prim算法，而不是克鲁斯卡尔
    //把0加入最大生成树
    for (CategoricalAttribute a = firstAtt + 1; a < noCatAtts_; a++) {
        maxWeight[a] = cmi[firstAtt][a]; //maxWeight首先是0到各个结点的距离
        if (cmi[firstAtt][a] > maxWeight[topCandidate])//找出与0距离最大的结点，作为下一个并入最小生成树的结点
            topCandidate = a;
        bestSoFar[a] = firstAtt; //目前各个属性的父结点设置为0
        available.insert(a);
    }
    //这个算法最精妙的地方在于，它把Prim算法先翻一遍lowcost 把权值最小的candidate并入最小生成树，再更新lowcost的2个for循环
    //并成了1个for循环，这个性质是符合完全图的，翻一遍lowcost 要检查未加入生成树的结点的个数，因为并入的candidate与所有
    //未加入生成树的结点相邻接，所以一样要比较未加入生成树的结点的个数，所以不像普通图一样，candidate只和几个结点相连，这样合并for循环就不合算了
    while (!available.empty()) {
        //candidate被加入最大生成树
        const CategoricalAttribute current = topCandidate;
        parents_[current] = bestSoFar[current];
        printf("%d->%d\t", bestSoFar[current], current);
        printf("%lf\n", cmi[bestSoFar[current]][current]);
        available.erase(current);

        if (!available.empty()) {//可能把candidate弄走了以后，所有的结点都在最大生成树中了，所以要再判断一下
            topCandidate = *available.begin();
            for (std::set<CategoricalAttribute>::const_iterator it = available.begin(); it != available.end(); it++) {
                //更新lowcost,如果current到it的权值比原来的大，就把lowcost更新，也更新父结点
                if (maxWeight[*it] < cmi[current][*it]) {
                    maxWeight[*it] = cmi[current][*it];
                    bestSoFar[*it] = current;
                }
                //当我更新完后，我要找到权值最大的
                if (maxWeight[*it] > maxWeight[topCandidate])
                    topCandidate = *it;
            }
        }

    }

    delete[] bestSoFar;
    delete[] maxWeight;

    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool tantest::trainingIsFinished() {
    return trainingIsFinished_;
}

void tantest::classify(const instance &inst, std::vector<double> &classDist) {
    //全局TAN分类
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

    //局部TAN
    //    crosstab<float> cmiLocal = crosstab<float>(noCatAtts_);
    //    getCondMutualInfloc(xxyDist_, cmiLocal, inst); //算I(xi;xj|Y) 用于构造最大生成树
    //
    //    std::vector<CategoricalAttribute> parents_loc; //局部的父子结构
    //    parents_loc.assign(noCatAtts_, NOPARENT);
    //
    //    CategoricalAttribute firstAtt = 0;
    //    parents_loc[firstAtt] = NOPARENT;
    //    float *maxWeight; //就是Prim里的lowcost 记录生成树到某个结点目前的最大权值
    //    CategoricalAttribute *bestParentSoFar; //记录目前给这个结点选的父结点
    //    CategoricalAttribute candidate = firstAtt; //这次要加入最大生成树的属性
    //    std::set<CategoricalAttribute> notInTree; //尚未加入生成树的属性
    //    safeAlloc(maxWeight, noCatAtts_); //其实就是 maxWeight=new float[noCatAtts_]
    //    safeAlloc(bestParentSoFar, noCatAtts_); //bestParentSoFar=new int[noCatAtts_]
    //
    //    maxWeight[firstAtt] = -std::numeric_limits<float>::max();
    //    for (CategoricalAttribute a = 1; a < noCatAtts_; a++) {
    //        maxWeight[a] = cmiLocal[firstAtt][a]; //maxWeight首先是0到各个结点的距离
    //        if (cmiLocal[firstAtt][a] > maxWeight[candidate])//找出与0距离最大的结点，作为下一个并入最小生成树的结点
    //            candidate = a;
    //        bestParentSoFar[a] = firstAtt; //目前各个属性的父结点设置为0
    //        notInTree.insert(a);
    //    }
    //    while (!notInTree.empty()) {
    //        const CategoricalAttribute current = candidate;
    //        parents_loc[current] = bestParentSoFar[current];
    //        notInTree.erase(current);
    //        if (!notInTree.empty())//可能把candidate弄走了以后，所有的结点都在最大生成树中了，所以要再判断一下
    //        {
    //            candidate = *notInTree.begin();
    //            for (std::set<CategoricalAttribute>::const_iterator it =
    //                    notInTree.begin(); it != notInTree.end(); it++) {
    //                if (maxWeight[*it] < cmiLocal[current][*it]) {
    //                    maxWeight[*it] = cmiLocal[current][*it];
    //                    bestParentSoFar[*it] = current;
    //                }
    //                //当我更新完后，我要找到权值最大的
    //                if (maxWeight[*it] > maxWeight[candidate])
    //                    candidate = *it;
    //            }
    //        }
    //    }
    //    delete[] bestParentSoFar;
    //    delete[] maxWeight;
    //
    //    //局部tan分类
    //    std::vector<double> classDistLocal(noClasses_, 0.0);
    //    for (CatValue y = 0; y < noClasses_; y++) {
    //        classDistLocal[y] = xxyDist_.xyCounts.p(y);
    //    }
    //
    //    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
    //        const CategoricalAttribute parent_loc = parents_loc[x1];
    //
    //        if (parent_loc == NOPARENT) {
    //            for (CatValue y = 0; y < noClasses_; y++) {
    //                classDistLocal[y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);
    //            }
    //        } else {
    //            for (CatValue y = 0; y < noClasses_; y++) {
    //                classDistLocal[y] *= xxyDist_.p(x1, inst.getCatVal(x1), parent_loc,
    //                        inst.getCatVal(parent_loc), y);
    //            }
    //        }
    //    }
    //    normalise(classDistLocal);
    if (noClasses_ >= 30000) {
        std::vector<CategoricalAttribute> orderGen;

        for (CategoricalAttribute a = 0; a < noClasses_; a++) {
            orderGen.push_back(a);
        }
        orderCmpClass cmpGen(&classDist);
        std::sort(orderGen.begin(), orderGen.end(), cmpGen);
        //    for (CatValue y = 0; y < noClasses_; y++) {
        //        printf("%d\t", orderGen[y]);
        //    }
        //    printf("\n");
        //    printf("Gen max3: %lf\t%lf\t%lf\n", classDist[orderGen[0]], classDist[orderGen[1]], classDist[orderGen[2]]);

        //locloc
        std::vector<double> classDistLocalloc(noClasses_, 0.0);
        for (CategoricalAttribute no = 0; no < 3; no++) {
            //        printf("%d:\n", orderGen[no]);
            crosstab<float> cmiLocalloc = crosstab<float>(noCatAtts_);
            getCondMutualInflocloc(xxyDist_, orderGen[no], cmiLocalloc, inst);

            std::vector<CategoricalAttribute> parents_locloc;
            parents_locloc.assign(noCatAtts_, NOPARENT);

            CategoricalAttribute firstAttloc = 0;
            parents_locloc[firstAttloc] = NOPARENT;
            float *maxWeightloc;
            CategoricalAttribute *bestParentSoFarloc;
            CategoricalAttribute candidateloc = firstAttloc;
            std::set<CategoricalAttribute> notInTreeloc;
            safeAlloc(maxWeightloc, noCatAtts_);
            safeAlloc(bestParentSoFarloc, noCatAtts_);

            maxWeightloc[firstAttloc] = -std::numeric_limits<float>::max();
            for (CategoricalAttribute a = 1; a < noCatAtts_; a++) {
                maxWeightloc[a] = cmiLocalloc[firstAttloc][a];
                if (cmiLocalloc[firstAttloc][a] > maxWeightloc[candidateloc])
                    candidateloc = a;
                bestParentSoFarloc[a] = firstAttloc;
                notInTreeloc.insert(a);
            }
            while (!notInTreeloc.empty()) {
                const CategoricalAttribute current = candidateloc;
                parents_locloc[current] = bestParentSoFarloc[current];
                notInTreeloc.erase(current);
                if (!notInTreeloc.empty())//
                {
                    candidateloc = *notInTreeloc.begin();
                    for (std::set<CategoricalAttribute>::const_iterator it =
                            notInTreeloc.begin(); it != notInTreeloc.end(); it++) {
                        if (maxWeightloc[*it] < cmiLocalloc[current][*it]) {
                            maxWeightloc[*it] = cmiLocalloc[current][*it];
                            bestParentSoFarloc[*it] = current;
                        }
                        //当我更新完后，我要找到权值最大的
                        if (maxWeightloc[*it] > maxWeightloc[candidateloc])
                            candidateloc = *it;
                    }
                }
            }
            delete[] bestParentSoFarloc;
            delete[] maxWeightloc;

            classDistLocalloc[orderGen[no]] = xxyDist_.xyCounts.p(orderGen[no]);

            for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
                const CategoricalAttribute parent_locloc = parents_locloc[x1];

                if (parent_locloc == NOPARENT) {
                    classDistLocalloc[orderGen[no]] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), orderGen[no]);
                } else {
                    classDistLocalloc[orderGen[no]] *= xxyDist_.p(x1, inst.getCatVal(x1), parent_locloc,
                            inst.getCatVal(parent_locloc), orderGen[no]);
                }
            }
        }
        normalise(classDistLocalloc);
        //    printf("finalloc: %lf\t%lf\t%lf\n", classDistLocalloc[orderGen[0]], classDistLocalloc[orderGen[1]], classDistLocalloc[orderGen[2]]);
        std::vector<double> classDist3(noClasses_, 0.0);
        double sum = classDist[orderGen[0]] + classDist[orderGen[1]] + classDist[orderGen[2]];
        classDist3[orderGen[0]] = classDist[orderGen[0]] / sum;
        classDist3[orderGen[1]] = classDist[orderGen[1]] / sum;
        classDist3[orderGen[2]] = classDist[orderGen[2]] / sum;
        //    printf("finalgen: %lf\t%lf\t%lf\n", classDist3[orderGen[0]], classDist3[orderGen[1]], classDist3[orderGen[2]]);
        //    printf("------------------------------------------\n");

        classDist.assign(noClasses_, 0.0);
        for (CatValue y = 0; y < noClasses_; y++) {
            classDist[y] = classDist3[y];
        }
        for (CatValue y = 0; y < noClasses_; y++) {
            classDist[y] += classDistLocalloc[y];
            classDist[y] = classDist[y] / 2;
        }
        normalise(classDist);
    } else {
        //locloc
        std::vector<double> classDistLocalloc(noClasses_, 0.0);
        for (CategoricalAttribute y = 0; y < noClasses_; y++) {
            printf("y = %d:\n", y);
            crosstab<float> cmiLocalloc = crosstab<float>(noCatAtts_);
            getCondMutualInflocloc(xxyDist_, y, cmiLocalloc, inst);
//            cmiLocalloc.print();
            std::vector<CategoricalAttribute> parents_locloc;
            parents_locloc.assign(noCatAtts_, NOPARENT);

            CategoricalAttribute firstAttloc = 0;
            parents_locloc[firstAttloc] = NOPARENT;
            printf("begin->%d\n", firstAttloc);
            float *maxWeightloc;
            CategoricalAttribute *bestParentSoFarloc;
            CategoricalAttribute candidateloc = firstAttloc;
            std::set<CategoricalAttribute> notInTreeloc;
            safeAlloc(maxWeightloc, noCatAtts_);
            safeAlloc(bestParentSoFarloc, noCatAtts_);

            maxWeightloc[firstAttloc] = -std::numeric_limits<float>::max();
            for (CategoricalAttribute a = 1; a < noCatAtts_; a++) {
                maxWeightloc[a] = cmiLocalloc[firstAttloc][a];
                if (cmiLocalloc[firstAttloc][a] > maxWeightloc[candidateloc])
                    candidateloc = a;
                bestParentSoFarloc[a] = firstAttloc;
                notInTreeloc.insert(a);
            }
            while (!notInTreeloc.empty()) {
                const CategoricalAttribute current = candidateloc;
                parents_locloc[current] = bestParentSoFarloc[current];
                printf("%d->%d\t", bestParentSoFarloc[current], current);
                printf("%lf\n", cmiLocalloc[bestParentSoFarloc[current]][current]);

                notInTreeloc.erase(current);
                if (!notInTreeloc.empty())//
                {
                    candidateloc = *notInTreeloc.begin();
                    for (std::set<CategoricalAttribute>::const_iterator it =
                            notInTreeloc.begin(); it != notInTreeloc.end(); it++) {
                        if (maxWeightloc[*it] < cmiLocalloc[current][*it]) {
                            maxWeightloc[*it] = cmiLocalloc[current][*it];
                            bestParentSoFarloc[*it] = current;
                        }
                        //当我更新完后，我要找到权值最大的
                        if (maxWeightloc[*it] > maxWeightloc[candidateloc])
                            candidateloc = *it;
                    }
                }
            }
            delete[] bestParentSoFarloc;
            delete[] maxWeightloc;

            printf("---------------------------------\n");




            classDistLocalloc[y] = xxyDist_.xyCounts.p(y);

            for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
                const CategoricalAttribute parent_locloc = parents_locloc[x1];

                if (parent_locloc == NOPARENT) {
                    classDistLocalloc[y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    classDistLocalloc[y] *= xxyDist_.p(x1, inst.getCatVal(x1), parent_locloc,
                            inst.getCatVal(parent_locloc), y);
                }
            }
        }
        normalise(classDistLocalloc);
        double max = -1;
        int pos = 0xFFFFFFFFUL;
        for (CatValue y = 0; y < noClasses_; y++) {
            if (max < classDist[y]) {
                max = classDist[y];
                pos = y;
            }
            //printf("%lf\t", classDistLocalloc[y]);
        }
        printf("General predict:%d\treal:%d\n", pos, inst.getClass());

        max = -1;
        pos = 0xFFFFFFFFUL;
        for (CatValue y = 0; y < noClasses_; y++) {
            if (max < classDistLocalloc[y]) {
                max = classDistLocalloc[y];
                pos = y;
            }
            //printf("%lf\t", classDistLocalloc[y]);
        }
        printf("local predict:%d\treal:%d\n", pos, inst.getClass());

        for (CatValue y = 0; y < noClasses_; y++) {
            classDist[y] += classDistLocalloc[y];
            classDist[y] = classDist[y] / 2;
        }
        normalise(classDist);
        max = -1;
        pos = 0xFFFFFFFFUL;
        for (CatValue y = 0; y < noClasses_; y++) {
            if (max < classDist[y]) {
                max = classDist[y];
                pos = y;
            }
            //printf("%lf\t", classDist[y]);
        }
        printf("G+L predict:%d\treal:%d\n", pos, inst.getClass());
        printf("=====================================\n");

    }
}