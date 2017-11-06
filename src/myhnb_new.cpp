#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <set>
#include <vector>
#include <math.h>

#include "myhnb_new.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"
using namespace std;

myhnb_new::myhnb_new(char*const*& argv, char*const* end):trainingIsFinished_(false)
{ 
  name_ = "Hidden Naive Bayes";

}


myhnb_new::~myhnb_new(void)
{
}

void  myhnb_new::getCapabilities(capabilities &c)
{
  c.setCatAtts(true);  // only categorical attributes are supported at the moment
}

void myhnb_new::reset(InstanceStream &is)
{
    instanceStream_=&is;
    const unsigned int noCatAtts=is.getNoCatAtts();
    noCatAtts_=noCatAtts;
    noClasses_=is.getNoClasses();
    
    /*初始数据结构空间*/
    dist_.reset(is);
    classDist_.reset(is);
    
    wij.resize(noCatAtts_);
    for(CategoricalAttribute i=0;i<noCatAtts_;++i)
    {
        wij[i].resize(noCatAtts_);
    }
    trainingIsFinished_ = false;

}


void myhnb_new::train(const instance &inst) {
  /*进行数据统计*/

    dist_.update(inst);
    classDist_.update(inst);
}


void myhnb_new::initialisePass() {
}


void myhnb_new::finalisePass() {
  //计算条件互信息

        crosstab<float> cmi=crosstab<float>(noCatAtts_);
        getCondMutualInf(dist_,cmi);
        
        if(verbosity>=3)
         {
                printf("\nConditional mutual information table\n");
                cmi.print();
        }
                //计算权值wij
         CategoricalAttribute i,j;
         std::vector<float> wij_denominator;
         wij_denominator.resize(noCatAtts_);

        for(i=0;i<noCatAtts_;++i)
        {
           for(j=0;j<noCatAtts_;++j)
           {
              wij_denominator[i]+=cmi[i][j];
           }
       
        }

         for(i=0;i<noCatAtts_;++i)
         {
            for(j=0;j<noCatAtts_;++j)
           {
               wij[i][j]=cmi[i][j]/wij_denominator[i];
           }

        }
     trainingIsFinished_ = true;

}

bool myhnb_new::trainingIsFinished()
{
    
    return  trainingIsFinished_;
    
    
}

/*
 * 该函数是每次单独对一个实例进行分类
 */
void myhnb_new::classify(const instance &inst, std::vector<double> &classDist) {

    std::vector<double> p_hidden;
  for (CatValue y = 0; y < noClasses_; y++)
  {
      double p_c = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);

      
      p_hidden.resize(noCatAtts_);
      p_hidden.assign(noCatAtts_,0);
      for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
      {
          for(CategoricalAttribute b=0;b < noCatAtts_;++b)
          {
              if(a!=b)
              {
                  p_hidden[a]+=((wij[a][b])*(dist_.p(a,inst.getCatVal(a),b,inst.getCatVal(b),y)));

              }
          }
      }
      for(CategoricalAttribute a=0;a<noCatAtts_;a++)
      {
          p_c*=p_hidden[a];
      }
      
      
      p_hidden.clear();
      
    /*最后求得p_c就是c(inst)*/

    assert(p_c >= 0.0f);
    /*classDist[y]存储的是实例属于每种类值的概率*/
    classDist[y] = p_c;
  }

  /*标准化类值概率，不影响类值概率大小关系*/
  normalise(classDist);
}

