/* Open source system for classification learning from very large data
** Class for an input stream of randomly sampled instances
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
#include "instanceSample.h"
#include <assert.h>
#include "utils.h"
#include "globals.h"

InstanceSample::InstanceSample(const unsigned int size) : targetSize(size), poolSize(0), cur(0),sumWeights_(0)
{ theSample.reserve(size);
}

InstanceSample::~InstanceSample(void)
{
}

void InstanceSample::setSource(InstanceStream &source) {
  source_ = &source;
  metaData_ = source.getMetaData();
  theSample.clear();
  poolSize = 0;
  sumWeights_=0;
  calcCDF_=true;
  CDF_.clear();
  sumWeightsForTargetInstance_=0;
  weightForTargetInstance_.clear();

  rewind();
}
void InstanceSample::rewind() {
  cur = 0;
}
// add the instance to the pool of vailable instances.
// instances are randomly sampled from the pool - do this as they are added
void InstanceSample::sample(instance &i) {
  ++poolSize;
  if (theSample.size() < targetSize) theSample.push_back(i);
  else {
    unsigned int index = rand(poolSize);
    if (index < targetSize) theSample[index] = i;
  }
}

// add the instance to the pool of vailable instances.
// instances are randomly sampled from the pool - do this as they are added
void InstanceSample::sampleWithWeights(instance &inst,std::vector<float> &weight,MTRand &randSampleInstance,MTRand &randReplaceInstance) {

	if (theSample.size() < targetSize) {
		//select the first targetSize instances

		sumWeights_ += weight[poolSize];
		weightForTargetInstance_.push_back(weight[poolSize]);
		theSample.push_back(inst);
	} else {

		//when targetSize instances have been selected, calculate the cdf for selecting instance for replacing
		if(calcCDF_==true)
		{
			assert(weightForTargetInstance_.size()==targetSize);

			//save the sum of weights of selected instances
			sumWeightsForTargetInstance_=sumWeights_;

			double sum=0;
			for(unsigned int k=0;k<targetSize;k++)
			{
				sum+=1-weightForTargetInstance_[k]/sumWeightsForTargetInstance_;
				CDF_.push_back(sum);
			}

			assert(CDF_.size()==targetSize);

			for(unsigned int k=0;k<targetSize;k++)
			{
				CDF_[k]/=sum;
			}
			calcCDF_=false;
		}

		double p = targetSize*weight[poolSize] /( sumWeights_+weight[poolSize] );
		//p=1;
		// with probability m*w_k/sum from {i=1} to {k}, select instance k and replace
		if (randSampleInstance() < p) {
			double pReplace=randReplaceInstance();

			unsigned int low=0;
			unsigned int up=targetSize-1;

			if(verbosity>=3)
			{
				printf("the probability to search: %f\n"
						"output the cdf:\n",pReplace);
				print(CDF_);
				printf("\n");

			}

			while(low<up)
			{
				unsigned int mid=(low+up)/2;
				if(pReplace<=CDF_[mid])
				{
					up=mid;
				}
				else
				{
					low=mid+1;

				}
			}
			assert(up == low);

			if(verbosity>=3)
			{
				printf("replace %u with %u\n",up,poolSize);
			}

			//update the sum of the weights of selected instances
			sumWeightsForTargetInstance_+=weight[poolSize]-weightForTargetInstance_[up];
			//update the weight of replaced
			weightForTargetInstance_[up]= weight[poolSize];
			//replace the instance
			theSample[up]=inst;


			//calculate the CDF again after replacing the instance

			CDF_.clear();
			double sum=0;
			for(unsigned int k=0;k<targetSize;k++)
			{
				sum+=1-weightForTargetInstance_[k]/sumWeightsForTargetInstance_;
				CDF_.push_back(sum);
			}

			assert(CDF_.size()==targetSize);

			for(unsigned int k=0;k<targetSize;k++)
			{
				CDF_[k]/=sum;
			}

		}
		sumWeights_ += weight[poolSize];

	}
	++poolSize;

}



InstanceCount InstanceSample::size() {
  return theSample.size();
}

/// advance, discarding the next instance in the stream.  Return true iff successful.
bool InstanceSample::advance() {
  ++cur;
  return cur < theSample.size();
}

/// get a pointer to the current instance.
/// Requires that there be a current instance, so must either check isAtEnd or the the most recent advance was successful.
instance* InstanceSample::current() {
  assert(cur < theSample.size());
  return &theSample[cur];
}

void InstanceSample::goTo(InstanceCount position) {
  assert(position>=0&&position <= theSample.size());

  cur=position-1;
}
/// advance to the next instance in the stream. Return true iff successful. @param inst the instance record to receive the new instance. 
bool InstanceSample::advance(instance &inst) {
  ++cur;
  if (cur >= theSample.size()) return false;
  else {
    inst = theSample[cur];
    return true;
  }
}

bool InstanceSample::isAtEnd() {
  return cur >= theSample.size();
}

/// return a string that gives a meaningful name for the stream
const char* InstanceSample::getName() {
  return "Sample";
}
