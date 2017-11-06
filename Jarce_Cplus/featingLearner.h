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

#include "learner.h"
#include "instanceStreamDiscretiser.h"

/**
<!-- globalinfo-start -->
 * Class for a bagged classifier.<br/>
 *
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 */
class FeatingLearner : public learner
{
public:
  
  /**
   * @param argv Options for the NB classifier
   * @param argc Number of options for NB
   */
  FeatingLearner(char*const*& argv, char*const* end);
  
  ~FeatingLearner(void);

  void getCapabilities(capabilities &c); 

  /**
   * trains the bagged committee of learners. 
   * 
   * @param is The training set
   */
  virtual void train(InstanceStream &is);

  /**
   * Calculates the class membership probabilities for the given test instance. 
   * 
   * @param inst The instance to be classified
   * @param classDist Predicted class probability distribution
   */
  virtual void classify(const instance &inst, std::vector<double> &classDist);
  
  
private:
  char const* learnerName_;                             ///< the name of the learner
  char*const* learnerArgv_;                             ///< the start of the arguments for the learner
  char*const* learnerArgEnd_;                           ///< the end of the arguments ot the learner
  unsigned int size_;                                   ///< the number of classifiers
  std::vector<std::vector<learner*> > catClassifiers_;  ///< the classifiers for the categorical attributes, indexed by attribute then value
  std::vector<std::vector<learner*> > numClassifiers_;  ///< the classifiers for the numeric attributes, indexed by attribute then discretised value
  bool useMajorityVoting_;                              ///< true iff the classifier uses majority voting
  InstanceStreamDiscretiser discretiser_;               ///< the discretiser used to create the values for numeric attributes
  capabilities capabilities_;
};

