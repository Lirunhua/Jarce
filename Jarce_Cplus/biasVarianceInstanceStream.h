#pragma once
#include "instanceStream.h"
#include "mtrand.h"

class biasVarianceInstanceStream: public InstanceStream {
public:
	biasVarianceInstanceStream(InstanceStream *source, const unsigned int noTraining, const unsigned int noTest,const unsigned int seed =
			0);
	~biasVarianceInstanceStream(void);

	// implementation of core InstanceStream methods
	void rewind();               ///< return to the first instance in the stream
	bool advance(); ///< advance, discarding the next instance in the stream.  Return true iff successful.
	bool advance(instance &inst); ///< advance to the next instance in the stream.  Return true iff successful. @param inst the instance record to receive the new instance.
	bool isAtEnd();         ///< true if we have advanced past the last instance
	InstanceCount size(); ///< the number of instances in the stream.  This may require a pass through the stream to determine so should be used only if absolutely necessary.

	//
	void setTraining(bool training);
private:
	InstanceStream* source_;       ///< the source stream
	MTRand randTest;
	MTRand randTrain;        ///< random number generator for selecting folds

	unsigned int noTrainingCases_;
	unsigned int noTestingCases_;

	InstanceCount trainRemaining_;
	InstanceCount testRemaining_;
	InstanceCount dataRemaining_;
	InstanceCount dataCount_;
	const unsigned int seed_;      ///< the random number seed

	bool training_; ///< true if the current pass is a training pass. If true, the stream returns instances from the training fold. If false, the stream returns instances from all folds other than the training fold.
	InstanceCount count_;  ///< a count of the number of instances in the stream
};
