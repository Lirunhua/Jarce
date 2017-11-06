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
 * Please report any bugs to Ana M. Martinez <anam.martinez@monash.edu>
 */

#pragma once

#include <limits>

#include "incrementalLearner.h"
#include "distributionTree.h"
#include "xxyDist.h"
#include "yDist.h"
#include "kdb.h"
#include "instanceStreamDiscretiser.h"



/**
<!-- globalinfo-start -->
 * Class for a k-dependence Bayesian classifier that (mdl-)discretises the numberic 
 * attributes conditioned on its parents (it has to be called jointly with an option 
 * for discretisation e.g. dmdl). This version takes more time, but it is less RAM 
 * demanding (writes on disk).<br/>
 * <br/>
 <!-- globalinfo-end -->
 *
 * @author Ana M. Martinez (anam.martinez@monash.edu)
 */


class kdbCondDisc :  public kdb
{
public:
  kdbCondDisc();
  kdbCondDisc(char*const*& argv, char*const* end);
  ~kdbCondDisc(void);

  void reset(InstanceStream &is);   ///< reset the learner prior to training
  void initialisePass();            ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
  void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
  void train(InstanceStream &is); 
  void finalisePass();              ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
  bool trainingIsFinished();        ///< true iff no more passes are required. updated by finalisePass()
  void getCapabilities(capabilities &c); ///< describes what kind of data the learner is able to handle

  virtual void classify(const instance &inst, std::vector<double> &classDist);
  
  void discretise(std::vector<NumValue> &vals, std::vector<CatValue> &valsDisc, std::vector<NumValue> &cuts); ///< discretises the numeric values in vector vals according to the cutpoint in vector cuts and writes them (in the same order) in vector valsDisc.
  
private:
  unsigned int noOrigCatAtts_ ;     ///< number of discrete attributes in the original dataset
  std::string dirTemp_;             ///< path of the directory for the temporal files
};
