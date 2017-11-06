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
** Please report any bugs to Ana M. Martinez <anam.martinez@monash.edu>
*/
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>
#ifdef __linux__
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "kdbCondDisc.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"
#include "instanceStreamDiscretiser.h"
#include "MDLDiscretiser.h"

std::string toString(float value)
{
  std::stringstream ss;
  ss << std::setprecision(std::numeric_limits<float>::digits10+2);
  ss << value;
  return ss.str();
}

template <typename T> std::string toString(T tmp)
{
    std::ostringstream out;
    out << tmp;
    return out.str();
}

kdbCondDisc::kdbCondDisc(char*const*& argv, char*const* end){
  name_ = "KDB-CondDisc";

  // defaults
  k_ = 1;
  
  // get arguments
  while (argv != end) {
    if (*argv[0] != '-') {
      break;
    }
    else if (argv[0][1] == 'k') {
      getUIntFromStr(argv[0]+2, k_, "k");
    }
    else if (argv[0][1] == 'r') {
      char *dirTemp = argv[0]+2;
      dirTemp_ = toString(dirTemp);
    }
    else {
      break;
    }

    name_ += argv[0];

    ++argv;
  }
  if(dirTemp_.size()==0){
    dirTemp_ = "temp";
#ifdef __linux__
    mkdir(dirTemp_.c_str(),0777);
#endif
  }
}

kdbCondDisc::~kdbCondDisc(void)
{
}


void  kdbCondDisc::getCapabilities(capabilities &c){
  c.setCatAtts(true);  
  c.setNumAtts(true);
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

void kdbCondDisc::reset(InstanceStream &is) {
  kdb::reset(is);
  noOrigCatAtts_ = static_cast<InstanceStreamDiscretiser::MetaData*>(static_cast<InstanceStreamDiscretiser&>(is).getMetaData())->getNoOrigCatAtts();
  #ifdef __linux__
  unsigned int dirNumber = 1;
  std::string dirTemp =  dirTemp_+"/temp"+toString(dirNumber);
  int status = mkdir(dirTemp_.c_str(),0777);
  while(status==-1){ //Different directory for different experiments
    dirNumber++;
    dirTemp =  dirTemp_+"/temp"+toString(dirNumber);
    status = mkdir(dirTemp.c_str(),0777);
  }
  dirTemp_ = dirTemp;
  #endif
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
void kdbCondDisc::train(const instance &inst) {
  
  instance instDisc(*instanceStream_);
  static_cast<InstanceStreamDiscretiser*>(instanceStream_)->discretiseInstance(inst, instDisc);
  
  if (pass_ == 1) {
    dist_.update(instDisc);
  }
  else {
    assert(pass_ == 2);

    for (CategoricalAttribute ca = 0; ca < noOrigCatAtts_; ca++) { //update counts as usual for the originally discrete attributes
      dTree_[ca].update(instDisc, ca, parents_[ca]);
    }
    for (CategoricalAttribute na = noOrigCatAtts_; na < noCatAtts_; na++) {
      if(parents_[na].size()==0)
        dTree_[na].update(instDisc, na, parents_[na]);             //Collect the xy distributions for the numeric attribute with no parents.
      else{
        std::vector<CategoricalAttribute> parents(parents_[na]);   //Remove the last parent and update, leave the last parent counts for the multiply disc. version
        parents.pop_back();
        dTree_[na].update(instDisc, na, parents);                  //For smoothing purposes.
      }
    }
    
    // Writes the numeric values for an attribute in a file of the form:
    // temp4_p1v3_p3v12_c0   
    // i.e. all the numeric values for attribute A4 conditioned on the parent-values A1=3 and A3=12 (being 3 and 12 discrete values, original or from the given discretization).
    // The class is indicated for the posterior mdl discretisation
    FILEtype *fdTree;
    for (NumericAttribute na = noOrigCatAtts_; na < noCatAtts_; na++) {
      if(parents_[na].size()>0){
        std::string fileName = dirTemp_+"/temp" + toString(na);
        for (std::vector<CategoricalAttribute>::const_iterator it = parents_[na].begin(); it != parents_[na].end(); it++) {
          fileName += "_p" + toString(*it) + "v" + toString(instDisc.getCatVal(*it));
        }
        fileName += "_c"+toString(inst.getClass());
        fdTree = fopen(fileName.c_str(), "a");
        std::string aux = toString(inst.getNumVal(na-noOrigCatAtts_));
        fputs((toString(inst.getNumVal(na-noOrigCatAtts_))+",").c_str(),fdTree);
        fclose(fdTree);
      }
    }
    classDist_.update(inst);
  }
}

void kdbCondDisc::train(InstanceStream &is){
  instance inst;
  
  testCapabilities(is);
  
  reset(is);

  while (!trainingIsFinished()) {
    initialisePass();
    is.rewind();
    while (static_cast<InstanceStreamDiscretiser&>(is).advanceNumeric(inst)) {
      train(inst);
    }
    finalisePass();
  }
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
void kdbCondDisc::initialisePass() {
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
void kdbCondDisc::finalisePass() {
  if (pass_ == 1) {
    // calculate the mutual information from the xy distribution
    std::vector<float> mi;  
    getMutualInformation(dist_.xyCounts, mi);
    
    if (verbosity >= 3) {
      printf("\nMutual information table\n");
      print(mi);
    }

    // calculate the conditional mutual information from the xxy distribution
    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    getCondMutualInf(dist_,cmi);
    
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
      miCmpClass cmp(&mi);

      std::sort(order.begin(), order.end(), cmp);

      if (verbosity >= 2) {
        printf("\n%s parents:\n", instanceStream_->getCatAttName(order[0]));
      }

      // proper KDB assignment of parents
      for (std::vector<CategoricalAttribute>::const_iterator it = order.begin()+1; it != order.end(); it++) {
        parents_[*it].push_back(order[0]);
        for (std::vector<CategoricalAttribute>::const_iterator it2 = order.begin()+1; it2 != it; it2++) {
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
  else if (pass_ == 2) {
#ifdef __linux__
    // STEP 1: Read files sequentially and find other files for the same attribute with same parent-values and different classes.
    //         Store all the values in vector vals, and the classes in vector classes.
    DIR *dp;
    struct dirent *dirp;
    //std::string dirName =  "temp"+toString(dirNumber);
    if((dp = opendir(dirTemp_.c_str()))) {  
      while (dirp = readdir(dp)) {
        if((dirp->d_name[0]!='.') && (dirp->d_name[strlen(dirp->d_name)-1] != '~')){
          
            std::vector<NumValue> vals;
            std::vector<CatValue> classes;
            std::vector<NumValue> cuts;  
            
            std::string fileName = dirp->d_name;  // next file to be read
            size_t classPos = fileName.find("c"); // position of the class in the string fileName 
            unsigned long  theClass = atoi(fileName.substr(classPos+1).c_str());  // the class label of the values in fileName
            
            // For example: temp4_p1v3_p3v12_c5   
            std::string path = dirTemp_+"/"+fileName;
            std::ifstream fdTree;  
            fdTree.open(path.c_str());
            if(fdTree){
              std::string value = "";

              while(std::getline(fdTree,value,','))
              {
                  vals.push_back(atof(value.c_str()));
                  classes.push_back(theClass);
              }
              fdTree.close();
              std::remove(path.c_str()); //delete file

              // For example: temp4_p1v3_p3v12_c0, temp4_p1v3_p3v12_c2, ...
              std::string fileName2 = fileName.substr(0,classPos-1).c_str();
              for (CatValue c = 0; c < instanceStream_->getNoClasses(); c++){
                  path = dirTemp_+"/"+fileName2+"_c"+toString(c);
                  fdTree.open(path.c_str());
                  if( (c != theClass) && fdTree){
                    while(std::getline(fdTree,value,','))
                    {
                      vals.push_back(atof(value.c_str()));
                      classes.push_back(c);
                    }
                    fdTree.close();
                    std::remove(path.c_str());
                  }else{
                    fdTree.close();
                  }
              }
              
              // STEP 2: The data for a combination of parent-values is stored in vector vals, now find the appropriate cuts
              MDLDiscretiser *theDiscretiser = new MDLDiscretiser();
              theDiscretiser->discretise(vals, classes, instanceStream_->getNoClasses(), cuts);
              delete theDiscretiser;
              
              // STEP 3: Discretise values according to cuts
              std::vector<CatValue> valsDisc;
              discretise(vals, valsDisc, cuts); //This could be done more efficiently by ordering vals first.
              
              // STEP 4: Update the probabilities and save cuts on the appropriate leaves of the dTree.
              //         We have to get the numeric attribute, and the parent-values combination
              size_t posNa = fileName.find("_"); 
              CategoricalAttribute na = atoi(fileName.substr(4,(posNa-4)).c_str());
              std::vector<CatValue> parentValues;  //The parent-values, e.g., {3,12} for temp4_p1v3_p3v12_c0
              for (unsigned int i = 0; i < parents_[na].size(); i++) { 
                 size_t posValue = fileName.find("v",posNa+1); 
                 size_t posValueEnd = fileName.find("_",posNa+1); 
                 CatValue v = atoi(fileName.substr(posValue+1,posValueEnd-(posValue+1)).c_str());
                 parentValues.push_back(v);
                 posNa = posValueEnd;
              }
              dTree_[na].update(valsDisc, na, classes, parents_[na], parentValues, cuts, noOrigCatAtts_);
            }
          }
      }
    } 
    closedir(dp);
    rmdir(dirTemp_.c_str());
    size_t posT = dirTemp_.find_last_of("/"); 
    dirTemp_ = dirTemp_.substr(0,posT);
#endif
  }
  ++pass_;
}

void kdbCondDisc::discretise(std::vector<NumValue> &vals, std::vector<CatValue> &valsDisc, std::vector<NumValue> &cuts) {
  for (std::vector<NumValue>::const_iterator it = vals.begin(); it != vals.end(); it++) {
      if (*it == MISSINGNUM) {
        valsDisc.push_back(cuts.size()+1);
      }
      else if (cuts.size() == 0) {
        valsDisc.push_back(0);
      }
      else if (*it > cuts.back()) {
        valsDisc.push_back(cuts.size());
      }
      else {
        unsigned int upper = cuts.size()-1;
        unsigned int lower = 0;

        while (upper > lower) {
          const unsigned int mid = lower + (upper-lower) / 2;

          if (*it <= cuts[mid]) {
            upper = mid;
          }
          else {
            lower = mid+1;
          }
        }

        assert(upper == lower);
        valsDisc.push_back(upper);
      }
  }
}


/// true iff no more passes are required. updated by finalisePass()
bool kdbCondDisc::trainingIsFinished() {
  return pass_ > 2;
}

void kdbCondDisc::classify(const instance& inst, std::vector<double> &posteriorDist) {
  // calculate the class probabilities in parallel
  // P(y)
  for (CatValue y = 0; y < noClasses_; y++) {
    posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
  }

  instance instDisc(*instanceStream_);
  static_cast<InstanceStreamDiscretiser*>(instanceStream_)->discretiseInstance(inst, instDisc);
  
  // P(x_i | x_p1, .. x_pk, y)
  for (CategoricalAttribute x = 0; x < noOrigCatAtts_; x++) {
      //Discretize inst appropriately according to multi conditional discretization for the numeric children and original discretization for the parents
    dTree_[x].updateClassDistribution(posteriorDist, x, instDisc);
  }
  for (CategoricalAttribute x = noOrigCatAtts_; x < noCatAtts_; x++) {
    if(parents_[x].size()==0)
      dTree_[x].updateClassDistribution(posteriorDist, x, instDisc);
    else
      dTree_[x].updateClassDistributionAndDiscAttValue(posteriorDist, x, instDisc, inst.getNumVal(x-noOrigCatAtts_), noOrigCatAtts_);
  }
  // normalise the results
  normalise(posteriorDist);
}




