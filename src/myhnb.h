/* 
 * File:   myhnb.h
 * Author: dragonriveryu
 *
 * Created on 2014年5月26日, 上午8:12
 */

#ifndef MYHNB_H
#define	MYHNB_H

#include <limits.h>
#include <vector>
#include "incrementalLearner.h"
#include "xxyDist.h"
#include "yDist.h"
//#include "crosstab.h"

class myhnb : public IncrementalLearner
{
public:
  
  /**
   * @param argv Options for the NB classifier
   * @param argc Number of options for NB
   */
  myhnb();
  myhnb(char*const*& argv, char*const* end);
  
  
  ~myhnb(void);
  
  void reset(InstanceStream &is);   ///< reset the learner prior to training
  void initialisePass();            ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
  void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
  void finalisePass();              ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
  bool trainingIsFinished();        ///< true iff no more passes are required. updated by finalisePass()
  void getCapabilities(capabilities &c); 

  /**
   * Calculates the class membership probabilities for the given test instance. 
   * 
   * @param inst The instance to be classified
   * @param classDist Predicted class probability distribution
   */
  virtual void classify(const instance &inst, std::vector<double> &classDist);
  
  
  /**
   * Calculates the class membership probabilities for the given test instance.
   * 
   * @return  The joint distribution for each individual x-value and the class
   */  
  //xyDist* getXyDist();
  
  
private:  

  unsigned int noCatAtts_;
  unsigned int noClasses_;
  xxyDist dist_;
  yDist classDist_;
  InstanceStream* instanceStream_;
  unsigned int pass_;
  
  std::vector<std::vector<float> > wij;     //保存每一对属性Ai和Aj之间计算读出来的权值
};


#endif	/* MYHNB_H */

