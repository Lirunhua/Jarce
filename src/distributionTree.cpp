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
#include "distributionTree.h"
#include "smoothing.h"
#include "utils.h"
#include <assert.h>

InstanceStream::MetaData const* dtNode::metaData_;

dtNode::dtNode(InstanceStream::MetaData const* meta, const CategoricalAttribute a) : att(NOPARENT), xyCount(meta->getNoValues(a),meta->getNoClasses()) {
  metaData_ = meta;
}

dtNode::dtNode(const CategoricalAttribute a) : att(NOPARENT), xyCount(metaData_->getNoValues(a),metaData_->getNoClasses()) {
}

//The parameter const CategoricalAttribute a is useless but intentionally added (used in kdb-condDisc)
dtNode::dtNode(const CategoricalAttribute a, unsigned int noValues) : att(NOPARENT), xyCount(noValues,metaData_->getNoClasses()) {
}

dtNode::dtNode() : att(NOPARENT) {
}

dtNode::~dtNode() {
}

void dtNode::init(InstanceStream::MetaData const* meta, const CategoricalAttribute a) {
  metaData_ = meta;
  att = NOPARENT;
  xyCount.assign(meta->getNoValues(a), meta->getNoClasses(), 0);
  children.clear();
}

void dtNode::clear() {
  xyCount.clear();
  children.clear();
  att = NOPARENT;
}

distributionTree::distributionTree() 
{
}

distributionTree::distributionTree(InstanceStream::MetaData const* metaData, const CategoricalAttribute att) : metaData_(metaData), dTree(metaData, att)
{
}

distributionTree::~distributionTree(void)
{
}

void distributionTree::init(InstanceStream const& stream, const CategoricalAttribute att)
{
  metaData_ = stream.getMetaData();
  dTree.init(metaData_, att);
}

void distributionTree::clear()
{
  dTree.clear();
}

dtNode* distributionTree::getdTNode(){
  return &dTree;
}

void distributionTree::update(const instance &i, const CategoricalAttribute a, const std::vector<CategoricalAttribute> &parents) {
  const CatValue y = i.getClass();
  const CatValue v = i.getCatVal(a);

  dTree.ref(v, y)++;

  dtNode *currentNode = &dTree;

  for (unsigned int d = 0; d < parents.size(); d++) { 

    const CategoricalAttribute p = parents[d];

    if (currentNode->att == NOPARENT || currentNode->children.empty()) {
      // children array has not yet been allocated
      currentNode->children.assign(metaData_->getNoValues(p), NULL);
      currentNode->att = p;
    }

    assert(currentNode->att == p);
    
    dtNode *nextNode = currentNode->children[i.getCatVal(p)];

    // the child has not yet been allocated, so allocate it
    if (nextNode == NULL) {
      currentNode = currentNode->children[i.getCatVal(p)] = new dtNode(a);
    }
    else {
      currentNode = nextNode;
    }

    currentNode->ref(v, y)++;
  }
}

  void distributionTree::update(const instance &i, const CategoricalAttribute a, const std::vector<CategoricalAttribute> &parents, NumValue attValue){
    const CatValue y = i.getClass();
    const CatValue v = i.getCatVal(a);

    dTree.ref(v, y)++;

    dtNode *currentNode = &dTree;

    int d = 0;
    for (d = 0; d < static_cast<int>(parents.size())-1; d++) { 

      const CategoricalAttribute p = parents[d];

      if (currentNode->att == NOPARENT || currentNode->children.empty()) {
        // children array has not yet been allocated
        currentNode->children.assign(metaData_->getNoValues(p), NULL);
        currentNode->att = p;
      }

      assert(currentNode->att == p);

      dtNode *nextNode = currentNode->children[i.getCatVal(p)];

      // the child has not yet been allocated, so allocate it
      if (nextNode == NULL) {
        currentNode = currentNode->children[i.getCatVal(p)] = new dtNode(a);
      }
      else {
        currentNode = nextNode;
      }

      currentNode->ref(v, y)++;
    }
    if(d < parents.size()){//store numeric values in the last leaf
      const CategoricalAttribute p = parents[d];

      if (currentNode->att == NOPARENT || currentNode->children.empty()) {
        // children array has not yet been allocated
        currentNode->children.assign(metaData_->getNoValues(p), NULL);
        currentNode->att = p;
      }

      assert(currentNode->att == p);

      dtNode *nextNode = currentNode->children[i.getCatVal(p)];

      // the child has not yet been allocated, so allocate it
      if (nextNode == NULL) {
        currentNode = currentNode->children[i.getCatVal(p)] = new dtNode(a,0);
        currentNode->numValues_.resize(metaData_->getNoClasses());
      }
      else {
        currentNode = nextNode;
      }
      currentNode->numValues_[y].push_back(attValue);
    }
  }

  void distributionTree::update(std::vector<CatValue> &valsDisc,  
                                const CategoricalAttribute a, std::vector<CatValue> &classes, const std::vector<CategoricalAttribute>  &parents, 
                                std::vector<CatValue> &parentValues, std::vector<NumValue> &cuts, unsigned int noOrigCatAtts){
      
      dtNode *currentNode = &dTree;
      dtNode *nextNode;
      
      unsigned int d = 0;
      for (d = 0; d < parents.size()-1; d++) { 
        const CategoricalAttribute p = parents[d];
        assert(currentNode->att == p);
        nextNode = currentNode->children[parentValues[d]];
        currentNode = nextNode;
      }
      //for the last parent
      if(parents.size()>0){
        
        const CategoricalAttribute p = parents[d];

        //Useful only for kdb-condDisc
        if (currentNode->att == NOPARENT || currentNode->children.empty()) {
          // children array has not yet been allocated
          currentNode->children.assign(metaData_->getNoValues(p), NULL);
          currentNode->att = p;
        }

        assert(currentNode->att == p);

        nextNode = currentNode->children[parentValues[d]];

        // the child has not yet been allocated, so allocate it
        if ((nextNode == NULL)) {//This for kdb-condDisc 
          currentNode = currentNode->children[parentValues[d]] = new dtNode(a, cuts.size() + 1 + metaData_->hasNumMissing(a - noOrigCatAtts));//this can be zero except for the last one
        }else if(nextNode->xyCount.getDim() == 0){//This for kdb-condDisc2
          nextNode->xyCount.resize(cuts.size() + 1 + metaData_->hasNumMissing(a - noOrigCatAtts),metaData_->getNoClasses());
          currentNode = nextNode;
        }else {
          currentNode = nextNode;
        }
        //only update counts on the leaves
        int i=0;
        for (std::vector<CategoricalAttribute>::const_iterator it = valsDisc.begin(); it != valsDisc.end(); it++, i++){
          currentNode->ref(*it, classes[i])++;
        }
      }
      //only store cuts on the leaves
      currentNode->cuts_ = cuts; //Only in the leaves. 
  }

// update classDist using the evidence from the tree about i
void distributionTree::updateClassDistribution(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i) {
  dtNode *dt = &dTree;
  CategoricalAttribute att = dTree.att;

  // find the appropriate leaf
  while (att != NOPARENT) {
    const CatValue v = i.getCatVal(att);
    dtNode *next = dt->children[v];
    if (next == NULL)
      break;
    dt = next;
    att = dt->att;
  }
  
 
  
  // sum over all values of the Attribute for the class to obtain count[y, parents]
  for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
    InstanceCount totalCount = dt->getCount(0, y);
    const unsigned int noOfVals = metaData_->getNoValues(a);

    for (CatValue v = 1; v < noOfVals; v++) {
      totalCount += dt->getCount(v, y);
    }

    classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount, noOfVals);
  }
}

// 根据第一个不同属性的不同，确定联合后验概率


// update classDist using the evidence from the tree about i
// require that at least minCount values be used for proability estimation
void distributionTree::updateClassDistribution(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i, InstanceCount minCount) {
  dtNode *dt = &dTree;
  CategoricalAttribute att = dTree.att;
  const CatValue parentVal = i.getCatVal(a);
  const CatValue noOfClasses = metaData_->getNoClasses();

  // find the appropriate leaf
  while (att != NOPARENT) {
    const CatValue v = i.getCatVal(att);
    dtNode *next = dt->children[v];
    if (next == NULL) break;

    // check that the next node has enough examples for this value;
    InstanceCount cnt = 0;
    for (CatValue y = 0; y < noOfClasses; y++) {
      cnt += next->getCount(parentVal, y);
      if (cnt >= minCount) goto next;
    }

    // break if the total count is < minCOunt
    break;

next:
    dt = next;
    att = dt->att;
  }

  // sum over all values of the Attribute for the class to obtain count[y, parents]
  const unsigned int noOfVals = metaData_->getNoValues(a);
  for (CatValue y = 0; y < noOfClasses; y++) {
    InstanceCount totalCount = dt->getCount(0, y);
    
    for (CatValue v = 1; v < noOfVals; v++) {
      totalCount += dt->getCount(v, y);
    }

    classDist[y] *= mEstimate(dt->getCount(parentVal, y), totalCount, noOfVals);
  }
}

void distributionTree::updateClassDistributionForK(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i, InstanceCount minCount, unsigned int k) {
  dtNode *dt = &dTree;
  CategoricalAttribute att = dTree.att;
  const CatValue parentVal = i.getCatVal(a);
  const CatValue noOfClasses = metaData_->getNoClasses();

  // find the appropriate leaf
  unsigned int depth = 0;
  while ( (att != NOPARENT) && (depth<k) ) { //We want to consider kdb k=k, we stop when the depth reached is equal to k
    depth++;
    const CatValue v = i.getCatVal(att);
    dtNode *next = dt->children[v];
    if (next == NULL) break;

    // check that the next node has enough examples for this value;
    InstanceCount cnt = 0;
    for (CatValue y = 0; y < noOfClasses; y++) {
      cnt += next->getCount(parentVal, y);
      if (cnt >= minCount) goto next;
    }

    // break if the total count is < minCOunt
    break;

next:
    dt = next;
    att = dt->att;
  }

  // sum over all values of the Attribute for the class to obtain count[y, parents]
  const unsigned int noOfVals = metaData_->getNoValues(a);
  for (CatValue y = 0; y < noOfClasses; y++) {
    InstanceCount totalCount = dt->getCount(0, y);
    
    for (CatValue v = 1; v < noOfVals; v++) {
      totalCount += dt->getCount(v, y);
    }

    classDist[y] *= mEstimate(dt->getCount(parentVal, y), totalCount, noOfVals);
  }
}

// update classDist using the evidence from the tree about i and deducting it at the same time (Pazzani's trick for loocv)
// require that at least 1 value (minCount = 1) be used for probability estimation
void distributionTree::updateClassDistributionloocv(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i) {
  dtNode *dt = &dTree;
  CategoricalAttribute att = dTree.att;

  // find the appropriate leaf
  while (att != NOPARENT) {
    const CatValue v = i.getCatVal(att);
    dtNode *next = dt->children[v];
    if (next == NULL) break;

    // check that the next node has enough examples for this value;
    InstanceCount cnt = 0;
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      cnt += next->getCount(i.getCatVal(a), y);
    }

    //In loocv, we consider minCount=1(+1), since we have to leave out i.
    if (cnt < 2) 
        break;

    dt = next;
    att = dt->att;
  }

  // sum over all values of the Attribute for the class to obtain count[y, parents]
  for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
    InstanceCount totalCount = dt->getCount(0, y);
    for (CatValue v = 1; v < metaData_->getNoValues(a); v++) {
      totalCount += dt->getCount(v, y);
    }    
    
    if(y!=i.getClass())
        classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount, metaData_->getNoValues(a));
    else
        classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y)-1, totalCount-1, metaData_->getNoValues(a));
  }
}

void distributionTree::updateClassDistributionloocv(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i, unsigned int k){
  dtNode *dt = &dTree;
  CategoricalAttribute att = dTree.att;

  // find the appropriate leaf
  unsigned int depth = 0;
  while ( (att != NOPARENT) && (depth<k) ) { //We want to consider kdb k=k, we stop when the depth reached is equal to k
    depth++;
    const CatValue v = i.getCatVal(att);
    dtNode *next = dt->children[v];
    if (next == NULL) break;

    // check that the next node has enough examples for this value;
    InstanceCount cnt = 0;
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      cnt += next->getCount(i.getCatVal(a), y);
    }

    //In loocv, we consider minCount=1(+1), since we have to leave out i.
    if (cnt < 2) 
        break;

    dt = next;
    att = dt->att;
  }

  // sum over all values of the Attribute for the class to obtain count[y, parents]
  for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
    InstanceCount totalCount = dt->getCount(0, y);
    for (CatValue v = 1; v < metaData_->getNoValues(a); v++) {
      totalCount += dt->getCount(v, y);
    }    
    
    if(y!=i.getClass())
        classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount, metaData_->getNoValues(a));
    else
        classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y)-1, totalCount-1, metaData_->getNoValues(a));
  }
}

// update classDist using the evidence from the tree about i
void distributionTree::updateClassDistributionAndDiscAttValue(std::vector<double> &classDist, const CategoricalAttribute a, 
                                                              const instance &i, NumValue attValue, unsigned int noOrigCatAtts) {
  dtNode *dt = &dTree;
  CategoricalAttribute att = dTree.att;

  bool smoothing = false;
  // find the appropriate leaf
  while (att != NOPARENT) {
    const CatValue v = i.getCatVal(att);
    dtNode *next = dt->children[v];
    if ((next == NULL) || (next->xyCount.getDim() == 0)){ //The second part required for kdb-condDisc2
      smoothing = true;
      break;
    }
    dt = next;
    att = dt->att;
  }
  
  if(smoothing){//Instead of P(x_i | x_p1, x_p2 ], x_p3, y) we use P(x_i | x_p1, x_p2, y), thus, we have to resort to the original discretisation
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      InstanceCount totalCount = dt->getCount(0, y);
      const unsigned int noOfVals = metaData_->getNoValues(a);

      for (CatValue v = 1; v < noOfVals; v++) {
        totalCount += dt->getCount(v, y);
      }
      classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount, noOfVals);
    }
  }else{
    // sum over all values of the Attribute for the class to obtain count[y, parents]
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      InstanceCount totalCount = dt->getCount(0, y);
      const unsigned int noOfVals = dt->cuts_.size() + 1 + metaData_->hasNumMissing(a - noOrigCatAtts);

      for (CatValue v = 1; v < noOfVals; v++) {
        totalCount += dt->getCount(v, y);
      }
      classDist[y] *= mEstimate(dt->getCount(discretise(attValue,dt->cuts_), y), totalCount, noOfVals);
    }
  }
}

// update classDist using the evidence from the tree about i
void distributionTree::updateClassDistributionAndDiscAttValue(std::vector<double> &classDist, const CategoricalAttribute a, 
                                                              const instance &i, NumValue attValue, unsigned int noOrigCatAtts, InstanceCount minCount) {
  dtNode *dt = &dTree;
  CategoricalAttribute att = dTree.att;
  const CatValue noOfClasses = metaData_->getNoClasses();
  const CatValue aVal = i.getCatVal(a);

  bool smoothing = false;
  // find the appropriate leaf
  while (att != NOPARENT) {
    const CatValue v = i.getCatVal(att);
    dtNode *next = dt->children[v];
    if ((next == NULL) || (next->xyCount.getDim() == 0)){ //The second part required for kdb-condDisc2
      smoothing = true;
      break;
    }
    // check that the next node has enough examples for this value;
    InstanceCount cnt = 0;
    for (CatValue y = 0; y < noOfClasses; y++) {
      cnt += next->getCount(aVal, y);
      if (cnt >= minCount) goto next;
    }
    // break if the total count is < minCOunt
    smoothing = true;
    break;

next:
    dt = next;
    att = dt->att;
  }
  
  if(smoothing){//Instead of P(x_i | x_p1, x_p2 , x_p3, y) we use P(x_i | x_p1, x_p2, y), thus, we have to resort to the original discretisation
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      InstanceCount totalCount = dt->getCount(0, y);
      const unsigned int noOfVals = metaData_->getNoValues(a);

      for (CatValue v = 1; v < noOfVals; v++) {
        totalCount += dt->getCount(v, y);
      }
      classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount, noOfVals);
    }
  }else{
    // sum over all values of the Attribute for the class to obtain count[y, parents]
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      InstanceCount totalCount = dt->getCount(0, y);
      const unsigned int noOfVals = dt->cuts_.size() + 1 + metaData_->hasNumMissing(a - noOrigCatAtts);

      for (CatValue v = 1; v < noOfVals; v++) {
        totalCount += dt->getCount(v, y);
      }
      classDist[y] *= mEstimate(dt->getCount(discretise(attValue,dt->cuts_), y), totalCount, noOfVals);
    }
  }
}

void dtNode::updateStats(CategoricalAttribute target, std::vector<CategoricalAttribute> &parents, unsigned int k, unsigned int depth, unsigned long long int &pc, double &apd, unsigned long long int &zc) {
  if (depth == parents.size()  || children.empty()) {
    for (CatValue v = 0; v < metaData_->getNoValues(target); v++) {
      pc++;

      apd += (depth-apd) / static_cast<double>(pc);

      for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
        if (getCount(v, y) == 0) zc++;
      }
    }
  }
  else {
    for (CatValue v = 0; v < metaData_->getNoValues(parents[depth]); v++) {
      if (children[v] == NULL) {
          unsigned long int pathsMissing = 1;
          
          for (unsigned int i = depth; i < parents.size(); i++) pathsMissing *= metaData_->getNoValues(parents[i]);

          pc += pathsMissing;

          apd += pathsMissing * ((depth-apd)/(pc-pathsMissing/2.0));

          for (CatValue tv = 0; tv < metaData_->getNoValues(target); tv++) {
            for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
              if (getCount(tv, y) == 0) zc++;
            }
          }
      }
      else {
        children[v]->updateStats(target, parents, k, depth+1, pc, apd, zc);
      }
    }
  }
}


void distributionTree::updateStats(CategoricalAttribute target, std::vector<CategoricalAttribute> &parents, unsigned int k, unsigned long long int &pc, double &apd, unsigned long long int &zc) {
  //dTree.updateStats(target, parents, parents.size(), 1, pc, apd, zc);
}
