/* Open source system for classification learning from very large data
 * Copyright (C) 2012 Geoffrey I Webb
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */

#pragma once

#include "incrementalLearner.h"
#include "xyDist.h"

/**
<!-- globalinfo-start -->
 * Class for sampling a portion of the data, it can be used to obtain a pre-discretised version of the dataset.<br/>
 * <br/>
 <!-- globalinfo-end -->
 *
 *
 *
 * @author Ana M Martinez (anam.martinez@monash.edu) and Shenglei cheng 
 */
class sample : public IncrementalLearner
{
public:
  
  /**
   * @param argv Options for the sample classifier
   * @param argc Number of options for sample
   */
  sample(char*const*& argv, char*const* end);
  
  
  ~sample(void);
  
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
  xyDist* getXyDist();
  
  
private:  
  bool trainingIsFinished_; ///< true iff the learner is trained
  InstanceCount  count;     ///< count of the number of instances seen so far
  InstanceCount  Cnt_;      ///< number of instances to sample (all of them by default, usefull for pre-discretisation) 
  FILE *f;                  ///< the output data file.
  char * filename_;         ///< the name of the output data file.
  unsigned int noCatAtts_;  ///< the number of categorical attributes.
  unsigned int noClasses_;  ///< the number of classes
  unsigned int noOrigCatAtts_ ;     ///< number of discrete attributes in the original dataset

};

