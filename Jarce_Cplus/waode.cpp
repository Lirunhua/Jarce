#include "waode.h"
#include <assert.h>
#include "utils.h"
#include <algorithm>
#include "correlationMeasures.h"
#include "globals.h"
#include "utils.h"
#include "crosstab.h"


//waode构造函数
waode::waode(char* const *& argv, char* const * end) {
    name_ = "WAODE";
    UsedAttrRatio = 0;
    weighted = false;
    minCount = 100;
    subsumptionResolution = false;
    selected = false;
    su_ = false;
    mi_ = false;
    ig_ = false;
    acmi_ = false;
    chisq_ = false;
    empiricalMEst_ = false;
    empiricalMEst2_ = false;

    correlationFilter_ = false;
    useThreshold_ = false;
    threshold_ = 0;
    factor_ = 1.0;

    useAttribSelec_ = false;

    chilect_ = false;

    attribSelected_ = 0;
    for (int i = 0; i < 100; i++) {
        fathercount[i] = 0;
    }
    
    //获取各种参数
    while (argv != end) {
        if (*argv[0] != '-') {
            break;
        } else if (streq(argv[0] + 1, "empirical")) {
            empiricalMEst_ = true;
        } else if (streq(argv[0] + 1, "empirical2")) {
            empiricalMEst2_ = true;
        } else if (streq(argv[0] + 1, "sub")) {
            subsumptionResolution = true;
        } else if (argv[0][1] == 'n') {
            getUIntFromStr(argv[0] + 2, minCount, "n");
        } else if (streq(argv[0] + 1, "w")) {
            weighted = true;
        } else if (streq(argv[0] + 1, "chilect")) {
            chilect_ = true;
        } else if (streq(argv[0] + 1, "selective")) {
            selected = true;
        } else if (streq(argv[0] + 1, "ig")) {
            selected = true;
            ig_ = true;
        } else if (streq(argv[0] + 1, "acmi")) {
            selected = true;
            acmi_ = true;
        } else if (streq(argv[0] + 1, "mi")) {
            selected = true;
            mi_ = true;
        } else if (argv[0][1] == 'a') {
            getUIntFromStr(argv[0] + 2, attribSelected_, "a");
            useAttribSelec_ = true;
        } else if (argv[0][1] == 'f') {
            unsigned int factor;
            getUIntFromStr(argv[0] + 2, factor, "f");
            factor_ = factor / 10.0;
            while (factor_ >= 1)
                factor_ /= 10;
        } else if (streq(argv[0] + 1, "su")) {
            selected = true;
            su_ = true;
        } else if (streq(argv[0] + 1, "cf")) {
            correlationFilter_ = true;
        } else if (argv[0][1] == 't') {
            unsigned int thres;
            getUIntFromStr(argv[0] + 2, thres, "threshold");
            threshold_ = thres / 10.0;
            while (threshold_ >= 1)
                threshold_ /= 10;
            useThreshold_ = true;
        } else if (streq(argv[0] + 1, "chisq")) {
            selected = true;
            chisq_ = true;
        } else {
            error("HNB_AODE does not support argument %s\n", argv[0]);
            break;
        }

        name_ += *argv;

        ++argv;
    }
    if (selected == true) {
        if (mi_ == false && su_ == false && chisq_ == false)
            chisq_ = true;
    }

    //设置训练没有结束
    trainingIsFinished_ = false;
}

//waode析构函数
waode::~waode(void) {
}

//不需要管，就是对属性的一种操作
void waode::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void waode::reset(InstanceStream &is) {
    
    //初始数据结构空间
    xxyDist_.reset(is);
    
    //添加
    xyDist_.reset(&is);
    
    
    trainingIsFinished_ = false;
    //没有被选择的属性数目
    inactiveCnt_ = 0;

    //得到属性的数目
    noCatAtts_ = is.getNoCatAtts();
    //得到类值的数目
    noClasses_ = is.getNoClasses();

    instanceStream_ = &is;

    //属性是否被当做父节点
    active_.assign(noCatAtts_, true);
    //属性是否被当做孩子节点
    chiactive_.assign(noCatAtts_, true);
    
    //添加
    wi.resize(noCatAtts_);
    
    
}

void waode::initialisePass() {

}

void waode::train(const instance &inst) {
    //根据每一个实例训练得结构
    xxyDist_.update(inst);
    
    //添加  
    xyDist_.update(inst);
}

/// true iff no more passes are required. updated by finalisePass()

bool waode::trainingIsFinished() {
    return trainingIsFinished_;
}

// creates a comparator for two attributes based on their
//relative value with the class,such as mutual information, symmetrical uncertainty

class valCmpClass {
public:

    valCmpClass(std::vector<float> *s) {
        val = s;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
        return (*val)[a] > (*val)[b];
    }

private:
    std::vector<float> *val;
};

//
int waode::getNextElement(std::vector<CategoricalAttribute> &order, CategoricalAttribute ca, unsigned int noSelected) {
    CategoricalAttribute c = ca + 1;
    while (active_[order[c]] == false && c < noSelected)
        c++;
    if (c < noSelected)
        return c;
    else
        return -1;
}

void waode::finalisePass() {

    /*添加*/
    //这一步需要计算条件互信息，计算的是属性变量和类变量的互信息
    getMutualInformation(xyDist_,wi);
    
 
    trainingIsFinished_ = true;
}
/*
 * 该函数每次只对一个实例进行分类
 */
void waode::classify(const instance &inst, std::vector<double> &classDist) {
    std::vector<bool> generalizationSet;  

    generalizationSet.assign(noCatAtts_, false);

    //compute the generalisation set and substitution set for
    //lazy subsumption resolution
    if (subsumptionResolution == true) {
        for (CategoricalAttribute i = 1; i < noCatAtts_; i++) {
            const CatValue iVal = inst.getCatVal(i);
            const InstanceCount countOfxi = xxyDist_.xyCounts.getCount(i, iVal);

            for (CategoricalAttribute j = 0; j < i; j++) {
                if (!generalizationSet[j]) {
                    const CatValue jVal = inst.getCatVal(j);
                    const InstanceCount countOfxixj = xxyDist_.getCount(i, iVal,
                            j, jVal);
                    const InstanceCount countOfxj = xxyDist_.xyCounts.getCount(
                            j, jVal);

                    if (countOfxj == countOfxixj && countOfxj >= minCount) {
                        //xi is a generalisation or substitution of xj
                        //once one xj has been found for xi, stop for rest j
                        generalizationSet[i] = true;
                        fathercount[j]++;
                        break;
                    } else if (countOfxi == countOfxixj
                            && countOfxi >= minCount) {
                        fathercount[i]++;
                        generalizationSet[j] = true;
                    }
                }
            }
        }
    }



    //classDist存储了样例属于每一个类的概率
    for (CatValue y = 0; y < noClasses_; y++)
        classDist[y] = 0;
   
    

    //用于判断样例的结构是否为朴素贝叶斯网络
    CatValue delta = 0; 

    sumOfwi=0;
    

    //fdarray相当于一个二维的数据结构
    fdarray<double> spodeProbs(noCatAtts_, noClasses_);
   ////所有的属性均设为未激活状态
    std::vector<bool> active(noCatAtts_, false);
    //计算P(Xi,Y)，每个属性节点和类节点作为父节点分别计算
    for (CatValue parent = 0; parent < noCatAtts_; parent++) {

        //discard the attribute that is not active or in generalization set
        if (!generalizationSet[parent]) {
            const CatValue parentVal = inst.getCatVal(parent);        
                //要求以Xi和Y为父节点的实例个数要大于0，这里的参数可以调整
                if (xxyDist_.xyCounts.getCount(parent, parentVal) > 0) {
                    delta++;
                    active[parent] = true;
                    for (CatValue y = 0; y < noClasses_; y++) {
                        spodeProbs[parent][y] =  xxyDist_.xyCounts.jointP(parent,inst.getCatVal(parent),y);
                    }
                } 
         
        }
    }

    //如果以Xi和Y为父节点的实例个数等于0，则可以将网络当做朴素贝叶斯网络
    //delta==0,说明各属性节点是完全独立，不相互依赖的。是朴树贝叶斯网络
    if (delta == 0) {
        nbClassify(inst, classDist, xxyDist_.xyCounts);
        return;
    }
    
  //计算P(Xj|Xi,Y)，不好理解，比较复杂
    for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
        if (!generalizationSet[x1]) {           
            const bool x1Active = active[x1];          
            
            for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {        
                if (!generalizationSet[x2]) {
                    const bool x2Active = active[x2];          
                    for (CatValue y = 0; y < noClasses_; y++) {
                        if (x1Active) {             
                          //为什么不用xxyDist.P()呢??
                          spodeProbs[x1][y] *=xxyDist_.jointP(x1,inst.getCatVal(x1),x2, inst.getCatVal(x2), y)/xxyDist_.xyCounts.jointP(x1, inst.getCatVal(x1), y);       
                        }
                        if (x2Active) {
                           spodeProbs[x2][y] *=xxyDist_.jointP(x1,inst.getCatVal(x1),x2, inst.getCatVal(x2), y)/xxyDist_.xyCounts.jointP(x2, inst.getCatVal(x2), y);    

                        }
                    }

                }
            }
        }
    }
    //这里应用到了权值，完成了WAODE算法
    for(CatValue y=0;y<noClasses_;y++)
    {
        for(CatValue parent=0;parent<noCatAtts_;parent++)
        {
            if(active[parent])
            {
                sumOfwi+=wi[parent];
                classDist[y]+=(wi[parent]*spodeProbs[parent][y]);
            }
        }
        classDist[y]=classDist[y]/sumOfwi;
    }
    /*************************************************************/
    //这部分没什么大用，只是为了看一下那些节点可以作为父节点使用
    float GenAttr = 0;
    for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
        if (active[i] == true) GenAttr++;
    }
    UsedAttrRatio += GenAttr / noCatAtts_;
   // printf("UsedAttrRatio is %f,\n", UsedAttrRatio);
    /*************************************************************/
    normalise(classDist);
}


//NB分类器
void waode::nbClassify(const instance &inst, std::vector<double> &classDist,
        xyDist &xyDist_) {

    for (CatValue y = 0; y < noClasses_; y++) {
        double p = xyDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
        // scale up by maximum possible factor to reduce risk of numeric underflow

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            p *= xyDist_.p(a, inst.getCatVal(a), y);
        }

        assert(p >= 0.0);
        classDist[y] = p;
    }
    normalise(classDist);
}
