#include "biasVarianceInstanceStream.h"
#include "globals.h"
#include "stdio.h"
biasVarianceInstanceStream::biasVarianceInstanceStream(InstanceStream *source,
		const unsigned int noTraining, const unsigned int noTest,
		const unsigned int seed) :
		source_(source), noTrainingCases_(noTraining), noTestingCases_(noTest), seed_(
				seed) {
	metaData_ = source->getMetaData();

	dataCount_ = source->size();

	training_ = true;
}

biasVarianceInstanceStream::~biasVarianceInstanceStream(void) {
}

void biasVarianceInstanceStream::setTraining(bool training) {
	training_ = training;
	rewind();
}
/// return to the first instance in the stream
void biasVarianceInstanceStream::rewind() {
	source_->rewind();
	randTrain.seed(seed_);
	randTest.seed(0);
	count_ = 0;

	trainRemaining_ = noTrainingCases_;
	testRemaining_ = noTestingCases_;
	dataRemaining_ = dataCount_;
}

/// advance, discarding the next instance in the stream.  Return true iff successful.
bool biasVarianceInstanceStream::advance() {
//	while (source_->advance()) {
//		if (rand_() % noOfFolds_ == fold_) {
//			// test instance
//			if (!training_) {
//				count_++;
//				return true;
//			}
//		} else {
//			// training instance
//			if (training_) {
//				count_++;
//				return true;
//			}
//		}
//	}
	printf("advance() has not been implemented\n");
	return false;
}

/// advance to the next instance in the stream.Return true iff successful. @param inst the instance record to receive the new instance. 
bool biasVarianceInstanceStream::advance(instance &inst) {
//

	if (training_) {

		while (!source_->isAtEnd() && trainRemaining_ > 0) {
			if (randTest()
					<= static_cast<double>(testRemaining_)
							/ static_cast<double>(dataRemaining_)) {
				// test instance
				if (source_->advance(inst)) {
					testRemaining_--;
					dataRemaining_--;
				}
			} else if (randTrain()
					<= static_cast<double>(trainRemaining_)
							/ static_cast<double>(dataRemaining_
									- testRemaining_)) {
				// training instance
				if (source_->advance(inst)) {

					trainRemaining_--;
					dataRemaining_--;
					return true;
				}
			} else {
				// discarded instance
				if (source_->advance(inst)) {
					dataRemaining_--;
				}
			}
		}
		return false;

	} else {
		while (!source_->isAtEnd() && testRemaining_ > 0) {
			if (randTest()
					<= static_cast<double>(testRemaining_)
							/ static_cast<double>(dataRemaining_)) {
				// test instance
				if (source_->advance(inst)) {
					testRemaining_--;
					dataRemaining_--;
					return true;
				}
			} else {
				// discarded instance
				if (source_->advance(inst)) {
					dataRemaining_--;
				}
			}
		}
	}
	return false;

}

/// true if we have advanced past the last instance
bool biasVarianceInstanceStream::isAtEnd() {
	return source_->isAtEnd();
}

/// the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.  The stream state is undefined after a call to size(), so a rewind shouldbe performed before the next advance.
InstanceCount biasVarianceInstanceStream::size() {
//	if (!isAtEnd()) {
//		instance inst(*this);
//
//		while (!isAtEnd())
//			advance(inst);
//	}
//
//	return count_;
	return noTrainingCases_ + noTestingCases_;
}

