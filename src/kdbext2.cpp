#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

//#include "kdbex.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"
#include "kdbext2.h"

kdbext2::kdbext2() : pass_(1) {
}

kdbext2::kdbext2(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "kdbext2";

    // defaults
    k_ = 2;

    // get arguments
    while (argv != end) {
        if (*argv[0] != '-') {
            break;
        } else if (argv[0][1] == 'k') {
            getUIntFromStr(argv[0] + 2, k_, "k");
        }
        else {
            break;
        }

        name_ += argv[0];

        ++argv;
    }
}

kdbext2::~kdbext2(void) {
}

void kdbext2::getCapabilities(capabilities &c) {
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

void kdbext2::reset(InstanceStream &is) {
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1

    // initialise distributions    
    xxxyDist_.reset(is);

    Accuracy_different.resize(noCatAtts_);
    SubCMI_label.resize(noCatAtts_);
    ModelOrder.resize(noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        SubCMI_label[a]=true;
        Accuracy_different[a] = 0;
        ModelOrder[a].resize(noCatAtts_);
    }
    
    dTree_.resize(noCatAtts);
    parents_.resize(noCatAtts);
    posteriorDist_different.resize(noCatAtts_);
    for (CatValue y = 0; y < noCatAtts_; y++) {
        posteriorDist_different[y].assign(noClasses_, 0.0);
    }

    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        for (CategoricalAttribute b = 0; b < noCatAtts; b++) {
            parents_[a].resize(noCatAtts);
            dTree_[a].resize(noCatAtts);
        }
    }
    for (CategoricalAttribute w = 0; w < noCatAtts; w++) {
        for (CategoricalAttribute b = 0; b < noCatAtts; b++) {
            parents_[w][b].clear();
            dTree_[w][b].init(is, b);
            ModelOrder[w][b].resize(noCatAtts_);
        }
    }
    dist_.reset(is);

    classDist_.reset(is);
    max_flag = 0;
    pass_ = 1;
    v=0;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

void kdbext2::train(const instance &inst) {
    xxxyDist_.update(inst);
   
    if (pass_ == 1) {
        dist_.update(inst);
        classDist_.update(inst);
    } 
    else if (pass_ == 2) {
        //    assert(pass_ == 2);

        //     xxyDist_.remove(inst);

        for (CategoricalAttribute w = 0; w < noCatAtts_; w++) {
            for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                dTree_[w][a].update(inst, a, parents_[w][a]);
            }
        }
    }
    else if (pass_ == 3) {
        for (CategoricalAttribute w = 0; w < noCatAtts_; w++) { 
            for (CatValue y = 0; y < noClasses_; y++) {
                posteriorDist_different[w][y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
            }
            // P(x_i | x_p1, .. x_pk, y)
            for (CategoricalAttribute x = 0; x < noCatAtts_; x++) {
                dTree_[w][x].updateClassDistribution(posteriorDist_different[w], x, inst);
            }
            normalise(posteriorDist_different[w]); 
            //********************************************************************
            if (indexOfMaxVal(posteriorDist_different[w]) == inst.getClass()) { 
                Accuracy_different[w]++;
            }
        }
        //     xxyDist_.add(inst);  
    }

}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void kdbext2::initialisePass() {
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void kdbext2::finalisePass() {
    if (pass_ == 1) {
        // calculate the mutual information from the xy distribution
        std::vector<float> mi;
        getMutualInformation(dist_.xyCounts, mi);
        for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
            float temp = mi[i];
        //    printf("the mutual information of %d is %f\n", i, temp);
        }
       // printf("\n");

        miCmpClass cmp(&mi);
        std::vector<CategoricalAttribute> order;
        order.clear();
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            order.push_back(a);
        }
        std::sort(order.begin(), order.end(), cmp);
     //   printf("order of MI\n");
        for (std::vector<CategoricalAttribute>::const_iterator i = order.begin(); i != order.end(); i++) {
            unsigned int u = *i;
     //       printf("%u\n", u);
        }
       v=* order.begin();
        crosstab<float> cmi = crosstab<float>(noCatAtts_);
        getCondMutualInf(dist_, cmi);
        //得到条件互信息
        for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
            for (CategoricalAttribute j = 0; j < noCatAtts_; j++) {
                float temp = cmi[i][j];
      //          printf("%f,", temp);
            }
      //      printf("\n");
        }
     // 条件互信息总和
        for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
            float sum_cmi = 0;
            for (CategoricalAttribute j = 0; j < noCatAtts_; j++) {
                sum_cmi += cmi[i][j];
            }           
        }
        unsigned int parentsOrder[noCatAtts_];
        float sum_kdbCMI[noCatAtts_];
        
        for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
            for (CategoricalAttribute j = 0; j < noCatAtts_; j++) {
                for (CategoricalAttribute k = 0; k < noCatAtts_; k++) {
                    ModelOrder[i][j][k] = -1;
                }
            }
        }
        for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
            sum_kdbCMI[i] = 0;
        }
        //**************************************************************************************
        for (unsigned int NodeOrder = 0; NodeOrder < noCatAtts_; NodeOrder++) {
            float temp_cmi[noCatAtts_][noCatAtts_];
            parentsOrder[0] = NodeOrder;       

            ModelOrder[NodeOrder][0][0] = NodeOrder;     


            for (int p = 1; p < noCatAtts_; p++) {    
                for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
                    for (CategoricalAttribute j = 0; j < noCatAtts_; j++) {
                        temp_cmi[i][j] = 0;
                    }
                }
                //               printf("hello\n");
                for (CategoricalAttribute other = 0; other < noCatAtts_; other++) {
                    bool exists = false;
                    for (CategoricalAttribute i = 0; i < p; i++) {
                        if (other == parentsOrder[i]) {
                            exists = true;
                        }
                    }
                    if (exists == false) {
                        for (CategoricalAttribute i = 0; i < p; i++) {
                            unsigned int temp_parent = parentsOrder[i];
                            temp_cmi[other][temp_parent] = cmi[other][temp_parent]; 
                        }
                    }
                }

                //****************************************************************************
                for (CategoricalAttribute other = 0; other < noCatAtts_; other++) {       
                    for (CategoricalAttribute loop = 0; loop < k_; loop++) {
                        float max = 0;
                        unsigned flag = 0;
                        for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
                            if (temp_cmi[other][i] >= max) {
                                max = temp_cmi[other][i];
                                flag = i;
                            }
                        }
                        temp_cmi[other][flag] = -temp_cmi[other][flag];
                    }
                }
                for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
                    for (CategoricalAttribute j = 0; j < noCatAtts_; j++) {
                        float tempCMI = temp_cmi[i][j];
                        //                      printf("%f; ", tempCMI);
                    }
                    //                  printf("\n");
                }
                for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
                    float k = 0;
                    for (CategoricalAttribute j = 0; j < noCatAtts_; j++) {
                        if (temp_cmi[i][j] < 0) {
                            k += temp_cmi[i][j];
                        }
                    }
                    temp_cmi[i][i] = k;
                    //                 printf("the sum of cmi of attribute %d is %f;  \n", i, k);
                }
                unsigned int max_flag;
                float max_k = 0;
                for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
                    if (-temp_cmi[i][i] >= max_k) {
                        max_k = -temp_cmi[i][i];
                        max_flag = i;
                    }
                }

                //              printf("the max flag is %u\n", max_flag);
                int NoOfChildren = 1;
                for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
                    if ((temp_cmi[max_flag][i] < 0) && (max_flag != i)) {
                        ModelOrder[NodeOrder][p][NoOfChildren] = i;
                        NoOfChildren++;
                        //                    parents_[max_flag].push_back(i);
                        sum_kdbCMI[NodeOrder] += cmi[max_flag][i];
                        //                     printf("%u 's parent is %d; ", max_flag, i);
                    }
                }
                //              printf("\n ");
                parentsOrder[p] = max_flag;
                ModelOrder[NodeOrder][p][0] = max_flag;
            }
        }
        float MaxSum_kdbCMI = 0;
        int flagOfMaxSum;
        for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
            float k = sum_kdbCMI[i];
            if (k >= MaxSum_kdbCMI) {
                MaxSum_kdbCMI = k;
                flagOfMaxSum = i;
            }
    //        printf("\n the sum CMI of %d 'th kdb is %f; ", i, k);
        }


        for (CategoricalAttribute w = 0; w < noCatAtts_; w++) {
     //       printf("the order of model %d is:\n ", w);
            for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
                int child = ModelOrder[w][i][0];//
    //            printf("child is %d, father is: ", child);
                for (CategoricalAttribute j = 1; j < k_ + 1; j++) {
                    int father = ModelOrder[w][i][j];//
                    if (father >= 0) {
      //                  printf("%d; ", father);
                        parents_[w][child].push_back(father);
                    }
                }
     //           printf("\n");
            }
        }

    } else if (pass_ == 3) {

        const InstanceCount totalCount = dist_.xyCounts.count; //鐤戦棶
        for (CategoricalAttribute w = 0; w < noCatAtts_; w++) {
            float k = Accuracy_different[w];
            printf(" zero-one loss of %d model is %f;\n", w, 1 - k / totalCount);
        }
        max_flag = indexOfMaxVal(Accuracy_different);
        printf("\n *********************************\n max is %d,total count is %d\n ***************************************\n", max_flag, totalCount);
    }
    ++pass_;
}



/// true iff no more passes are required. updated by finalisePass()

bool kdbext2::trainingIsFinished() {
    return pass_ > 2;
}

void kdbext2::classify(const instance& inst, std::vector<double> &posteriorDist) {

    //    for (CategoricalAttribute w = 0; w < noCatAtts_; w++) {//

    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
    }
   
    float P_dist[noCatAtts_];
    for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
        P_dist[i] = 1;
    }
    max_flag = 0;
    float max_value=0;
    CategoricalAttribute w=v;
//    for (CategoricalAttribute w = 0; w < noCatAtts_; w++) {
//        int first = ModelOrder[w][0][0];
//        int second = ModelOrder[w][1][0];
//        int third = ModelOrder[w][2][0];
//
//        P_dist[w] = xxxyDist_.jointP(first, inst.getCatVal(first), second, inst.getCatVal(second), third, inst.getCatVal(third));
        for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {
            int child = ModelOrder[w][i][0]; //0位置是内容是, 第w个模型的第i个子变量
            int FirstFatherOf_i_Attr = ModelOrder[w][i][1];
            int SecondFatherOf_i_Attr = ModelOrder[w][i][2];
            if (FirstFatherOf_i_Attr < 0) {
                P_dist[w] *= xxxyDist_.xxyCounts.xyCounts.p(child, inst.getCatVal(child));
            } else if (SecondFatherOf_i_Attr < 0) {
                P_dist[w] *= xxxyDist_.xxyCounts.jointP(child, inst.getCatVal(child), FirstFatherOf_i_Attr, inst.getCatVal(FirstFatherOf_i_Attr)) / xxxyDist_.xxyCounts.xyCounts.p(FirstFatherOf_i_Attr, inst.getCatVal(FirstFatherOf_i_Attr));
            } else
                P_dist[w] *= xxxyDist_.jointP(child, inst.getCatVal(child), FirstFatherOf_i_Attr, inst.getCatVal(FirstFatherOf_i_Attr), SecondFatherOf_i_Attr, inst.getCatVal(SecondFatherOf_i_Attr)) / xxxyDist_.xxyCounts.jointP(FirstFatherOf_i_Attr, inst.getCatVal(FirstFatherOf_i_Attr), SecondFatherOf_i_Attr, inst.getCatVal(SecondFatherOf_i_Attr));
        }
        if (P_dist[w] >= max_value) {
            max_flag = w;
            max_value = P_dist[w];
        }
//    }
    
    
    
    // P(x_i | x_p1, .. x_pk, y)
   
    
    for (CategoricalAttribute x = 0; x < noCatAtts_; x++) {
        dTree_[max_flag][x].updateClassDistribution(posteriorDist, x, inst);
    }
//    int label=inst.getClass();
//    P_dist_all+=log10(posteriorDist[label]);
//    printf("the final joint probability is %f,\n",P_dist_all);
    normalise(posteriorDist); //
    //********************************************************************     
}

