/* Open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
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
#include <queue>

#include "kdbExt.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"
#include "ALGLIB_specialfunctions.h"
#include "crosstab.h"

kdbExt::kdbExt(char*const*& argv, char*const* end) {
  name_ = "KDB-EXT";

  // defaults
  k_ = 1;
  tan_ = false;
  randomOrder_ = false;
  randomParents_ = false;
  chisq_ = false;
  holm_ = false;
  chisqParents_ = false;
  bestLinks_ = false;
  orderByValue_ = false;
  discrimVals_ = false;
  selective_ = false;
  selectiveMCC_ = false;
  selectiveTest_ = false;
  selectiveWeighted_ = false;
  selectiveK_ = false;
  selectiveSampling_ = false;
  su_ = false;    
  orderBysu_ = false;
  orderBycmi_ = false;
  cmiParents_ = false;
  minCount_ = 0;
  trainSize_ = 0;
  sampleSize_ = 50;
  srand(sampleSize_);
  
  // get arguments
  while (argv != end) {
    if (*argv[0] != '-') {
      break;
    }
    else if (argv[0][1] == 'k') {
      getUIntFromStr(argv[0]+2, k_, "k");
    }
    else if (argv[0][1] == 'm') {
     getUIntFromStr(argv[0]+2, minCount_, "m");
    }
    else if (streq(argv[0]+1, "tan")) {
      tan_ = true;
    }
    else if (streq(argv[0]+1, "randomOrder")) {
      randomOrder_ = true;
    }
    else if (streq(argv[0]+1, "randomParents")) {
      randomParents_ = true;
    }
    else if (streq(argv[0]+1, "chisq")) {
      chisq_ = true;
    }
    else if (streq(argv[0]+1, "chisq-holm")) {
      chisq_ = true;
      holm_ = true;
    }
    else if (streq(argv[0]+1, "chisq-parents")) {
      chisqParents_ = true;
    }
    else if (streq(argv[0]+1, "bestLinks")) {
      bestLinks_ = true;
    }
    else if (streq(argv[0]+1, "orderByValue")) {
      orderByValue_ = true;
    }
    else if (streq(argv[0]+1, "discrimVals")) {
      discrimVals_ = true;
    }
    else if (streq(argv[0]+1, "selective")) {
      selective_ = true;
    }
    else if (streq(argv[0]+1, "selectiveMCC")) {
      selectiveMCC_ = true;
      selective_ = true;
    }
    else if (streq(argv[0]+1, "selectiveTest")) {
      selectiveTest_ = true;
      selective_ = true;
    }
    else if (streq(argv[0]+1, "selectiveWeighted")) {
      selectiveWeighted_ = true;
      selective_ = true;
    }
    else if (streq(argv[0]+1, "selectiveK")) {
      selectiveK_ = true;
      selective_ = true;
    }
    else if (streq(argv[0]+1, "selectiveSampling")) {
      selectiveSampling_ = true;
      selective_ = true;
    }
    else if (argv[0][1] == 's') {
     getUIntFromStr(argv[0]+2, sampleSize_, "s");
    }
    else if (streq(argv[0]+1, "su")) {
      su_ = true;
    }
    else if (streq(argv[0]+1, "orderBysu")) {
      orderBysu_ = true;
    }
    else if (streq(argv[0]+1, "orderBycmi")) {
      orderBycmi_ = true;
    }
    else if (streq(argv[0]+1, "cmiParents")) {
      cmiParents_ = true;
    }
    else {
      break;
    }

    name_ += argv[0];

    ++argv;
  }
}

kdbExt::~kdbExt(void)
{
}

void  kdbExt::getCapabilities(capabilities &c){
  c.setCatAtts(true);  // only categorical attributes are supported at the moment
}

void kdbExt::reset(InstanceStream &is) {
  kdb::reset(is);
  
  order_.clear();
  active_.assign(noCatAtts_, true);
  foldLossFunct_.assign(noCatAtts_+1,0.0); //1 more for the prior
  binomialTestCounts_.assign(noCatAtts_+1,0); //1 more for the prior
  sampleSizeBinomTest_.assign(noCatAtts_+1,0);//1 more for the prior
  if (selectiveMCC_) {
      if(selectiveK_){
        xtab_.resize(k_);
//        TPallK_.resize(k_);
//        FPallK_.resize(k_);
//        TNallK_.resize(k_);
//        FNallK_.resize(k_);
        for(int i=0; i< k_; i++){
          xtab_[i].resize( noCatAtts_+1, crosstab<InstanceCount>(noClasses_));
          for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            xtab_[i][a] = crosstab<InstanceCount>(noClasses_);
          }
//          TPallK_[i].assign(noCatAtts_+1,0); //1 more for the prior
//          FPallK_[i].assign(noCatAtts_+1,0); //1 more for the prior
//          TNallK_[i].assign(noCatAtts_+1,0); //1 more for the prior
//          FNallK_[i].assign(noCatAtts_+1,0); //1 more for the prior
        }
      }
      else{
          xtab_.resize(1);
          xtab_[0].resize( noCatAtts_+1, crosstab<InstanceCount>(noClasses_));
          for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            xtab_[0][a] = crosstab<InstanceCount>(noClasses_);
          }          
//        TP_.assign(noCatAtts_+1,0); //1 more for the prior
//        FP_.assign(noCatAtts_+1,0); //1 more for the prior
//        TN_.assign(noCatAtts_+1,0); //1 more for the prior
//        FN_.assign(noCatAtts_+1,0); //1 more for the prior
      }
  }
  if(selectiveK_){
    foldLossFunctallK_.resize(k_);
    for(int i=0; i< k_; i++){
      foldLossFunctallK_[i].assign(noCatAtts_+1,0);
    }
  }
  inactiveCnt_ = 0;
  trainSize_ = 0;
  sampleSizeDec_ = sampleSize_;
  if(cmiParents_)
    xxxyDist_.reset(is);
  
}

void kdbExt::train(const instance &inst) {
  if (pass_ == 1) {
    // in the first pass collect the xxy distribution
    if(cmiParents_)
      xxxyDist_.update(inst);
    else
      dist_.update(inst);
    trainSize_++; // to calculate the RMSE for each LOOCV
  }
  else if(pass_ == 2){
    // on the second pass collect the distributions to the k-dependence classifier
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
      dTree_[a].update(inst, a, parents_[a]);
    }
    classDist_.update(inst);
  }else{
      assert(pass_ == 3); //only for selective KDB
      if(selectiveK_){
        for(int k=0; k< k_; k++){
          std::vector<double> posteriorDist(noClasses_);
          //Only the class is considered
          for (CatValue y = 0; y < noClasses_; y++) {
            posteriorDist[y] = classDist_.ploocv(y, inst.getClass());//Discounting inst from counts
          }
          normalise(posteriorDist);
          const CatValue trueClass = inst.getClass();
          
          if (selectiveMCC_) {
            const CatValue prediction = indexOfMaxVal(posteriorDist);
            xtab_[k][noCatAtts_][trueClass][prediction]++;
//            if (trueClass == 0) {
//              if (posteriorDist[0] >= 0.5) {
//                TPallK_[k][noCatAtts_]++;
//              }
//              else {
//                FPallK_[k][noCatAtts_]++;
//              }
//            }
//            else if (posteriorDist[0] < 0.5) {
//              TNallK_[k][noCatAtts_]++;
//            }
//              else FNallK_[k][noCatAtts_]++;
          }else{
            const double error = 1.0-posteriorDist[trueClass];
            foldLossFunctallK_[k][noCatAtts_] += error*error;
          }

          for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); 
                                                                 it != order_.end(); it++){
              dTree_[*it].updateClassDistributionloocv(posteriorDist, *it, inst, k+1);//Discounting inst from counts
              normalise(posteriorDist);
              
              if (selectiveMCC_) {
                const CatValue prediction = indexOfMaxVal(posteriorDist);
                xtab_[k][*it][trueClass][prediction]++;
//                if (trueClass == 0) {
//                  if (posteriorDist[0] >= 0.5) {
//                    TPallK_[k][*it]++;
//                  }
//                  else {
//                    FPallK_[k][*it]++;
//                  }
//                }
//                else if (posteriorDist[0] < 0.5) {
//                  TNallK_[k][*it]++;
//                }
//                else FNallK_[k][*it]++;
              }else{
                const double error = 1.0-posteriorDist[trueClass];
                foldLossFunctallK_[k][*it] += error*error;
              }
              
          }
        }
      }else{
          if(selectiveSampling_){
              //double f = rand()/(double)RAND_MAX;
              //double g = sampleSizeDec_/(double)trainSizeDec_;
            //if(f <= g){   
            if(sampledInstaces[trainSizeDec_]){
                sampleSizeDec_--;
                std::vector<double> posteriorDist(noClasses_);
                std::vector<double> errorsAtts; // Store the att errors for this instance (needed for selectiveTest)
                errorsAtts.assign(noCatAtts_+1,0.0);
                //Only the class is considered
                for (CatValue y = 0; y < noClasses_; y++) {
                  posteriorDist[y] = classDist_.ploocv(y,inst.getClass());//Discounting inst from counts
                }
                normalise(posteriorDist);
                const CatValue trueClass = inst.getClass();
                const double error = 1.0-posteriorDist[trueClass];
                if (selectiveWeighted_) {
                  foldLossFunct_[noCatAtts_] += (1.0-prior_[trueClass])*error*error;
                }
                else {
                  foldLossFunct_[noCatAtts_] += error*error;
                }
                errorsAtts[noCatAtts_] = error;

                for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); 
                                                                       it != order_.end(); it++){
                    dTree_[*it].updateClassDistributionloocv(posteriorDist, *it, inst);//Discounting inst from counts
                    normalise(posteriorDist);
                    const double error = 1.0-posteriorDist[trueClass];

                    if (selectiveWeighted_) {
                      foldLossFunct_[*it] += (1.0-prior_[trueClass]) * error*error;
                    }
                    else {
                      foldLossFunct_[*it] += error*error;
                    }
                    errorsAtts[*it] = error;
                }
            }
            trainSizeDec_--;
          }else{ //Proper kdb selective
            std::vector<double> posteriorDist(noClasses_);
            std::vector<double> errorsAtts; // Store the att errors for this instance (needed for selectiveTest)
            errorsAtts.assign(noCatAtts_+1,0.0);
            //Only the class is considered
            for (CatValue y = 0; y < noClasses_; y++) {
              posteriorDist[y] = classDist_.ploocv(y,inst.getClass());//Discounting inst from counts
            }
            normalise(posteriorDist);
            const CatValue trueClass = inst.getClass();
            const double error = 1.0-posteriorDist[trueClass];
            if (selectiveWeighted_) {
              foldLossFunct_[noCatAtts_] += (1.0-prior_[trueClass])*error*error;
            }
            else {
              foldLossFunct_[noCatAtts_] += error*error;
            }
            errorsAtts[noCatAtts_] = error;
            if (selectiveMCC_) {
                const CatValue prediction = indexOfMaxVal(posteriorDist);
                xtab_[0][noCatAtts_][trueClass][prediction]++;
//              if (trueClass == 0) {
//                if (posteriorDist[0] >= 0.5) {
//                  TP_[noCatAtts_]++;
//                }
//                else {
//                  FP_[noCatAtts_]++;
//                }
//              }
//              else if (posteriorDist[0] < 0.5) {
//                TN_[noCatAtts_]++;
//              }
//              else FN_[noCatAtts_]++;
            }

            for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); 
                                                                   it != order_.end(); it++){
                dTree_[*it].updateClassDistributionloocv(posteriorDist, *it, inst);//Discounting inst from counts
                normalise(posteriorDist);
                const double error = 1.0-posteriorDist[trueClass];

                if (selectiveWeighted_) {
                  foldLossFunct_[*it] += (1.0-prior_[trueClass]) * error*error;
                }
                else {
                  foldLossFunct_[*it] += error*error;
                }
                errorsAtts[*it] = error;
                if (selectiveMCC_) {
                  const CatValue prediction = indexOfMaxVal(posteriorDist);
                  xtab_[0][*it][trueClass][prediction]++;
//                  if (trueClass == 0) {
//                    if (posteriorDist[0] >= 0.5) {
//                      TP_[*it]++;
//                    }
//                    else {
//                      FP_[*it]++;
//                    }
//                  }
//                  else if (posteriorDist[0] < 0.5) {
//                    TN_[*it]++;
//                  }
//                  else FN_[*it]++;
                }
            }

            if(selectiveTest_){
              double allerror = errorsAtts[order_[noCatAtts_-1]];
              //Only considering the class
              if ( (errorsAtts[noCatAtts_] - allerror) < 0.00001){//Draws are not counted
              }
              else if ( errorsAtts[noCatAtts_] < allerror ){
                  binomialTestCounts_[noCatAtts_]++;
                  sampleSizeBinomTest_[noCatAtts_]++;
              }
              else if ( errorsAtts[noCatAtts_] > allerror )//Draws are not counted
                  sampleSizeBinomTest_[noCatAtts_]++;
              //For the rest of the attributes
              for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); 
                                                                     it != order_.end()-1; it++){
                if( fabs(errorsAtts[*it]- allerror) < 0.00001){//Draws are not counted
                }
                else if( errorsAtts[*it] < allerror ){
                  binomialTestCounts_[*it]++;
                  sampleSizeBinomTest_[*it]++;
                }
                else if( errorsAtts[*it] > allerror ){
                  sampleSizeBinomTest_[*it]++;
                }
              }
            }
          }
      }
  }
}

/// true iff no more passes are required. updated by finalisePass()
bool kdbExt::trainingIsFinished() {
  if(selective_)
    return pass_ > 3;
  else
    return pass_ > 2;
}

void kdbExt::classify(const instance &inst, std::vector<double> &posteriorDist) {
  const unsigned int noClasses = noClasses_;

  for (CatValue y = 0; y < noClasses; y++) {
    posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
  }

  for (CategoricalAttribute x = 0; x < noCatAtts_; x++) {
    if (active_[x]) {
      if (minCount_) {
        if(selectiveK_) {
          dTree_[x].updateClassDistributionForK(posteriorDist, x, inst, minCount_, bestK_);
        }else {
          dTree_[x].updateClassDistribution(posteriorDist, x, inst, minCount_);
        }
      }
      else if(selectiveK_){
             //dTree_[x].updateClassDistributionForK(posteriorDist, x, inst, bestK_);
      }else{
             dTree_[x].updateClassDistribution(posteriorDist, x, inst);
      }
    }
  }

  normalise(posteriorDist);
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

class linkRec {
public:
  linkRec(const CategoricalAttribute aa, const CategoricalAttribute ab, const float v) : 
                                                                a1(aa), a2(ab), val(v) {}

  CategoricalAttribute a1;
  CategoricalAttribute a2;
  float val;
  const bool operator <(const linkRec& x) const {
    return (val < x.val);
  }
  struct CompGreater {
    bool operator()(const linkRec &l1, const linkRec &l2) {
      return l1.val > l2.val;
    }
  };
};


void kdbExt::finalisePass() {
  if (pass_ == 1) {
    
    std::vector<float> mi;  
    crosstab<float> cmi = crosstab<float>(noCatAtts_);  //CMI(X;Y|C) = H(X|C) - H(X|Y,C) -> cmi[X][Y]
    crosstab<float> acmi = crosstab<float>(noCatAtts_); //ACMI(X;C|Y) = H(X|Y) - H(X|C,Y) -> acmi[X][Y]
    std::vector<crosstab<float> > mcmi;                 //MCMI(X;C|Y,Z) = H(X|Y,Z) - H(X|C,Y,Z) -> mcmi[X][Y][Z]
              
    if(su_){
      getSymmetricalUncert(dist_.xyCounts, mi);
      getCondSymmUncert(dist_, cmi);
    }
    else if(orderBysu_){
      getSymmetricalUncert(dist_.xyCounts, mi);
      getCondMutualInf(dist_,cmi);
    }
    else if(discrimVals_){
      getMutualInformation(dist_.xyCounts, mi);
      getErrorDiff(dist_,cmi);
    }
    else if(orderBycmi_ && !cmiParents_){
      getMutualInformation(dist_.xyCounts, mi);
      getBothCondMutualInf(dist_, cmi, acmi);
    }
    else if(cmiParents_){
      getMutualInformation(xxxyDist_.xxyCounts.xyCounts, mi);
      for (unsigned int i = 0; i < noCatAtts_; i++) {
          mcmi.push_back(crosstab<float>(noCatAtts_)); 
        }
      getMultCondMutualInf(xxxyDist_, mcmi); 
      if(orderBycmi_){
        getBothCondMutualInf(xxxyDist_.xxyCounts, cmi, acmi);
      }
      else{
        getCondMutualInf(xxxyDist_.xxyCounts, cmi);
      }    
      xxxyDist_.clear();
    }
    else{ //Proper KDB
      getMutualInformation(dist_.xyCounts, mi);
      getCondMutualInf(dist_,cmi);
    }

    if (selectiveWeighted_) {
      for (CatValue y = 0; y < dist_.getNoClasses(); ++y) {
        prior_.push_back(dist_.xyCounts.p(y));
      }
    }
    
    if(!chisq_){
      dist_.clear();
    }
    
    if (verbosity >= 3) {
      printf("\nMutual information table\n");
      print(mi);
      putchar('\n');
    }
    
    if (verbosity >= 3) {
      printf("\nConditional mutual information table\n");
      cmi.print();
    }
    
    // sort the attributes on MI with the class
    

    if (bestLinks_) {
      error("Bestlinks not currently supported!");
      // select the best links first and then try to assign as many as possible of them
      std::priority_queue<linkRec, std::vector<linkRec>, linkRec::CompGreater > links;
      //const unsigned int target = k * meta->noAttributes - (k+1) * (k/2); // this sets target to exactly the maximum number of parent relationships that can be created
      const unsigned int target = k_ * noCatAtts_; // this sets target to slightly more than the maximum number of parent relationships that can be created

      // find the target number of best parent relationships
      for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        order_.push_back(a);
        for (CategoricalAttribute a2 = a + 1; a2 < noCatAtts_; a2++) {
          linkRec thisLink(a, a2, cmi[a][a2]);

          if (links.size() < target) links.push(thisLink);
          else {
            if (links.top() < thisLink) {
              links.pop();
              links.push(thisLink);
            }
          }
        }
      }

      // find out how many links each att has and sort the atts from the smallest number of links to the greatest
      std::vector<linkRec> linkvec;
      std::vector<unsigned int> linkCnt;
      linkCnt.assign(noCatAtts_, 0);
      std::vector<unsigned int> orderReverseIndex;
      orderReverseIndex.assign(noCatAtts_, 0);

      while (!links.empty()) {
        linkCnt[links.top().a1]++;
        linkCnt[links.top().a2]++;
        linkvec.push_back(links.top());
        links.pop();
      }

      IndirectCmpClass2<unsigned int, float> cmp(linkCnt, mi);
      std::sort(order_.begin(), order_.end(), cmp);

      // create an index from order to att
      for (unsigned int i = 0; i < order_.size(); i++) {
        orderReverseIndex[order_[i]] = i;
      }

      linkCnt.assign(noCatAtts_, 0);

      // establish the parent relationships
      unsigned int count = 0; // count of the number of parent relationships

      for (std::vector<linkRec>::reverse_iterator it = linkvec.rbegin(); 
                                                        it != linkvec.rend(); it++) {
        const CategoricalAttribute a1 = it->a1;
        const CategoricalAttribute a2 = it->a2;

        if (orderReverseIndex[a1] < orderReverseIndex[a2]) {
          if (linkCnt[a2] < k_) {
            parents_[a2][linkCnt[a2]++] = a1;
            count++;
          }
        }
        else {
          if (linkCnt[a1] < k_) {
            parents_[a1][linkCnt[a1]++] = a2;
            count++;
          }
        }
      }

      // some attribute may have less than k links - the following code, when complete, will fill them in
      //for (unsigned int i = 1; i < meta->noAttributes; i++) {
      //  const attribute a = orderReverseIndex[i];
      //  if (meta->attTypes[a] == categorical) {
      //    const unsigned int extra = min(i, k - linkCnt[a]);

      //    if (extra > 0) {
      //    }
      //  }
      //}

      if (verbosity >= 3) {
        printf("%u parent relationships, %u not utilised\n", count, k_ * noCatAtts_ - (k_)
                                                                       * (k_/2) - count);
      }
    }
    else { // Not BestLinks
      for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        order_.push_back(a);
      }

      if (!order_.empty()) {
        if (randomOrder_) {
          randomise(order_);
        }
        else if (orderByValue_) {
          // use the total value of all potential links to order the attributes
          float *val;
          allocAndClear(val, noCatAtts_);

          for (CategoricalAttribute a1 = 0; a1 < noCatAtts_; a1++) {
            for (CategoricalAttribute a2 = 0; a2 < a1; a2++) {
              val[a1] += cmi[a1][a2];
              val[a2] += cmi[a1][a2];
            }
          }

          IndirectCmpClass<float> cmp(val);
          std::sort(order_.begin(), order_.end(), cmp);
          
          delete []val;
        }
        else if(orderBycmi_ ){
          unsigned int bestParent = 0;
          std::vector<CategoricalAttribute> newOrder;
          
          //For the selection of the first attribute, only MI(A,C) is used
          //bestParent = std::max_element(mi.begin(),mi.end())-mi.begin();
          //newOrder.push_back(order_[bestParent]);
          //order_.erase(order_.begin()+bestParent);
          
          //But we prefer to have them ordered in order to solve draws.
          miCmpClass cmp(&mi);
          std::sort(order_.begin(), order_.end(), cmp);
          newOrder.push_back(order_[0]);
          order_.erase(order_.begin());
          
          
          while(order_.size()!=1){
            float maxInf = -1.0;
            unsigned int i=0, bestIt;
            for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); 
                                                                  it != order_.end(); it++,i++){//candidate parents
              float minInf = std::numeric_limits<float>::max();
              for (std::vector<CategoricalAttribute>::const_iterator it2 = newOrder.begin(); 
                                                                  it2 != newOrder.end(); it2++){ 
                if(acmi[*it][*it2]<minInf)
                  minInf = acmi[*it][*it2];
              }
              if(minInf > maxInf){
                maxInf = minInf;
                bestParent = *it;
                bestIt = i;
              }              
            }        
            newOrder.push_back(bestParent);
            order_.erase(order_.begin()+bestIt);
          }
          newOrder.push_back(order_[0]); //last element
          order_ = newOrder;         
          
        }
        else { // normal ordering
          miCmpClass cmp(&mi);

          std::sort(order_.begin(), order_.end(), cmp);
          if(verbosity>=3){
              printf("Att. Order (mi): ");
              for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); 
                                                                 it != order_.end(); it++){
                  printf("%s,",instanceStream_->getCatAttName(*it));
              }     
              putchar('\n');
          }
        }

        if (chisq_) {
          std::vector<CategoricalAttribute> newOrder;

          for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); 
                                                                 it != order_.end(); it++){
            CategoricalAttribute a = *it;
            const unsigned int rows = instanceStream_->getNoValues(a);
            
            
            if (rows < 2) {
              active_[a] = false;
              inactiveCnt_++;
            }
            else {
              const unsigned int cols = noClasses_;
              InstanceCount *tab;
              allocAndClear(tab, rows * cols);

              for (CatValue r = 0; r < rows; r++) {
                for (CatValue c = 0; c < cols; c++) {
                  tab[r*cols+c] += dist_.xyCounts.getCount(a, r, c);
                }
              }

              double critVal;
              
              if (holm_) critVal = 0.05 / (order_.end() - it);
              else critVal = 0.05;
              
              if (chiSquare(tab, rows, cols) > critVal) {
                if (verbosity >= 3) printf("%s suppressed by chisq test against class\n", 
                                           instanceStream_->getCatAttName(a));
                active_[a] = false;
                inactiveCnt_++;
                if (holm_) {
                  delete []tab;
                  break;
                }
              }
              else {
                newOrder.push_back(*it);
              }

              delete []tab;
            }
          }

          order_ = newOrder;
        }

        if (!order_.empty()) {
          if (verbosity >= 3) {
            printf("\n%s parents:\n", instanceStream_->getCatAttName(order_[0]));
          }

          if (randomParents_) {
            // allocate parents randomly
            std::vector<CategoricalAttribute>::const_iterator it = order_.begin();

            std::vector<CategoricalAttribute> thisParents;
            thisParents.push_back(*it);

            while (++it != order_.end()) {
              if (verbosity >= 3) {
                printf("%s parents: ", instanceStream_->getCatAttName(*it));
              }
              randomise(thisParents);
              for (unsigned int i = 0; i < min(k_, 
                                     static_cast<unsigned int>(it - order_.begin())); i++){
                parents_[*it].push_back(thisParents[i]);
                if (verbosity >= 3) {
                  printf("%s ", instanceStream_->getCatAttName(parents_[*it].back()));
                }
              }

              if (verbosity >= 3) {
                putchar('\n');
              }

              thisParents.push_back(*it);
            }
          }
          else if(cmiParents_){
              
            //Automatic parents when order is < k_
            unsigned int i=0;
            for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin()+1;  
                                                                 i != k_; it++,i++){
              for (std::vector<CategoricalAttribute>::const_iterator 
                                                it2 = order_.begin(); it2 != it; it2++) {
                 parents_[*it].push_back(*it2);
              }
              if (verbosity >= 2) {
                printf("%s parents: ", instanceStream_->getCatAttName(*it));
                for (unsigned int i = 0; i < parents_[*it].size(); i++) {
                  printf("%s ", instanceStream_->getCatAttName(parents_[*it][i]));
                }
                putchar('\n');
              }
            }     
            
            //Parent selection for the rest of the attributes
            for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin()+k_+1;  
                                                                 it != order_.end(); it++){
                
              //First parent decision for the rest of the attributes
              float maxInf = -1.0;
              unsigned int bestParent = 0;
              for (std::vector<CategoricalAttribute>::const_iterator 
                                                it2 = order_.begin(); it2 != it; it2++) {
                if(cmi[*it][*it2]>maxInf){
                  maxInf = cmi[*it][*it2];
                  bestParent = *it2;
                }                  
              }
              parents_[*it].push_back(bestParent);
              
              //Decision for the rest of the parents of the attributes
              bool noMoreParents = false;
              while((parents_[*it].size()!=k_) && !noMoreParents){
                float maxInf = 0.0;
                unsigned int bestParent = std::numeric_limits<unsigned int>::max()-1;
                for (std::vector<CategoricalAttribute>::const_iterator 
                                                  it2 = order_.begin(); it2 != it; it2++) {
                    //Explore all the candidate parents (have not been yet picked)
                    if((std::find(parents_[*it].begin(),parents_[*it].end(),*it2) == parents_[*it].end())){
                      float minInf = std::numeric_limits<float>::max();
                      for (std::vector<CategoricalAttribute>::const_iterator it3 = parents_[*it].begin();  
                                                                     it3 != parents_[*it].end(); it3++){
                        if(mcmi[*it][*it2][*it3]<minInf)
                          minInf = mcmi[*it][*it2][*it3];
                      }
                      if(minInf > maxInf){
                        maxInf = minInf;
                        bestParent = *it2;
                      }  
                    }
                }
                if(bestParent!=(std::numeric_limits<unsigned int>::max()-1))
                  parents_[*it].push_back(bestParent);
                else{
                  noMoreParents = true;
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
          else {
            // proper KDB assignment of parents
            //if (verbosity >= 2)
            //    printf("%s parents: \n", instanceStream_->getCatAttName(order_[0]));
            for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin()+1;  
                                                                 it != order_.end(); it++){
              parents_[*it].push_back(order_[0]);
              for (std::vector<CategoricalAttribute>::const_iterator 
                                                it2 = order_.begin()+1; it2 != it; it2++) {
                if (chisqParents_) {
                  const unsigned int rows = instanceStream_->getNoValues(*it);

                  if (rows < 2) continue;

                  const unsigned int cols = instanceStream_->getNoValues(*it2);
                  InstanceCount *tab;
                  allocAndClear(tab, rows * cols);

                  for (CatValue r = 0; r < rows; r++) {
                    for (CatValue c = 0; c < cols; c++) {
                      for (CatValue y = 0; y < noClasses_; y++) {
                        tab[r*cols+c] += dist_.getCount(*it, r, *it2, c, y);
                      }
                    }
                  }

                  if (chiSquare(tab, rows, cols) > 0.05) {
                    delete []tab;
                    continue;
                  }

                  delete []tab;
                }

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
                    for (unsigned int j = parents_[*it].size()-1; j > i; j--) {
                      parents_[*it][j] = parents_[*it][j-1];
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
        }
      }
    }
  }
  else if(pass_ == 3) {//only for selective KDB
    int bestatt;

    if(selectiveK_){
      if (selectiveMCC_){
        for (unsigned int k=0; k<k_;k++) {
          //foldLossFunctallK_[k][noCatAtts_] = -calcBinaryMCC(TPallK_[k][noCatAtts_], FPallK_[k][noCatAtts_], TNallK_[k][noCatAtts_], FNallK_[k][noCatAtts_]);
          foldLossFunctallK_[k][noCatAtts_] = -calcMCC(xtab_[k][noCatAtts_]);
          for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
            const unsigned int i = *it;
            //foldLossFunctallK_[k][i] = -calcBinaryMCC(TPallK_[k][i], FPallK_[k][i], TNallK_[k][i], FNallK_[k][i]);
            foldLossFunctallK_[k][i] = -calcMCC(xtab_[k][i]);
          }
        }
      }else{//Proper kdb selective (RMSE)      
        for (unsigned int k=0; k<k_;k++) {
          for (unsigned int att=0; att<noCatAtts_+1;att++) {
            foldLossFunctallK_[k][att] = sqrt(foldLossFunctallK_[k][att]/trainSize_);
          }
        }
      }
      if(verbosity>=3){
        if(selectiveMCC_){
          printf("MCC: \n");
          for (unsigned int k=0; k<k_;k++) {
            printf("k = %d : ",k+1);
            for (unsigned int att=0; att<noCatAtts_;att++) {
              printf("%.3f,", -foldLossFunctallK_[k][att]);
            }
            printf("%.3f(class)\n", -foldLossFunctallK_[k][noCatAtts_]);
          }
        }
        else{
          printf("RMSE: \n");
          for (unsigned int k=0; k<k_;k++) {
            printf("k = %d : ",k+1);
            for (unsigned int att=0; att<noCatAtts_;att++) {
              printf("%.3f,", foldLossFunctallK_[k][att]);
            }
            printf("%.3f(class)\n", foldLossFunctallK_[k][noCatAtts_]);
          }
        }
      }

      double min = foldLossFunctallK_[0][noCatAtts_];
      bestatt = noCatAtts_;
      bestK_ = 0; //naive Bayes
      //foldSumRMSE_[k][noCatAtts_] should be the same for all k
      for (unsigned int k=0; k<k_;k++) {
        for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
          if(foldLossFunctallK_[k][*it] < min){
            min = foldLossFunctallK_[k][*it];
            bestatt = *it;
            bestK_ = k+1;
          }
        }
      }
    }
    else{//proper selective

      if (selectiveMCC_){
        //foldLossFunct_[noCatAtts_] = -calcBinaryMCC(TP_[noCatAtts_], FP_[noCatAtts_], TN_[noCatAtts_], FN_[noCatAtts_]);
        foldLossFunct_[noCatAtts_] = -calcMCC(xtab_[0][noCatAtts_]);
        for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
          const unsigned int i = *it;
          //foldLossFunct_[i] = -calcBinaryMCC(TP_[i], FP_[i], TN_[i], FN_[i]);
          foldLossFunct_[i] = -calcMCC(xtab_[0][i]);
        }
      }
      else{
        for (unsigned int att=0; att<foldLossFunct_.size();att++) {
          foldLossFunct_[att] = sqrt(foldLossFunct_[att]/trainSize_);
        }
      }
      if(verbosity>=3){
        if (selectiveMCC_)
          printf("MCC: ");
        else
          printf("RMSE: ");
        for (unsigned int att=0; att<foldLossFunct_.size()-1;att++) {
          printf("%.3f,", foldLossFunct_[att]);
        }
        printf("%.3f(class)", foldLossFunct_[noCatAtts_]);
      }

      if(verbosity>=3)
        putchar('\n');      

      //Find the best attribute in order (to resolve ties in the best way possible)
      //It is the class only by default
      double min = foldLossFunct_[noCatAtts_];
      bestatt = noCatAtts_;
      for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
        if(foldLossFunct_[*it] < min){
          min = foldLossFunct_[*it];
          bestatt = *it;
        }
      }
    //int bestatt = std::min_element(foldSumRMSE.begin(), foldSumRMSE.end())-foldSumRMSE.begin();
    }

    if(selectiveTest_){
      //H_0 -> There is no difference between selecting until bestatt or taking all the attributes
      if (alglib::binomialcdistribution(binomialTestCounts_[bestatt], 
        sampleSizeBinomTest_[bestatt],0.5) > 0.05)
        //Complemented binomial distribution: calculates p(x>binomialTestCounts[bestatt])
        //H_0 is accepted
        bestatt = order_[noCatAtts_-1];
    }

    bool erase;
    if(bestatt==noCatAtts_)
      erase = true;
    else 
      erase = false;     
    for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
        if(erase)
        {
          active_[*it] = false;
          inactiveCnt_++;
        }
        else if(*it==bestatt)
          erase=true;                  
    }
    if(verbosity>=2){
      printf("Number of features selected is: %d out of %d\n",noCatAtts_-inactiveCnt_, noCatAtts_);
      if(selectiveK_)
        printf("best k is: %d\n",bestK_);
    }
  }else{
    assert(pass_ == 2);
    trainSizeDec_ = trainSize_;
    //Select sampleSize_ instances from the trainSize_ instances
    if(sampleSize_ >= trainSize_)
      sampledInstaces.assign(trainSize_,true);
    else{
      sampledInstaces.assign(trainSize_,false);
      for(int i=0; i<sampleSize_; i++){
         unsigned int index = rand(trainSize_);
         sampledInstaces[index] = true;
      }        
    }
  }
  ++pass_;
}



void kdbExt::printClassifier() {
  // there is a bug in distributionTree::updateStats, so this functionality is currently disabled
  //double apd = 0;
  //unsigned long long int pc = 0;
  //unsigned long long int zc = 0;
  //for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
  //    dTree_[a].updateStats(a, parents_[a], k_, pc, apd, zc);
  //}
  //printf("\nTotal number of paths: %llu\nAverage completed path depth: %f\nNumber of 0 counts: %llu\n", pc, apd, zc);
}
