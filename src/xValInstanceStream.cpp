#include "xValInstanceStream.h"
#include "stdio.h"
#include <iostream>
using namespace std;

XValInstanceStream::XValInstanceStream(InstanceStream *source, const unsigned int noOfFolds, const unsigned int seed, const unsigned int flag)
: source_(source), noOfFolds_(noOfFolds), seed_(seed), flag_(flag) {
    metaData_ = source->getMetaData();
    startSubstream(0, true, false);
    divide = 5;
}

XValInstanceStream::~XValInstanceStream(void) {
}

/// start training or testing for a new fold

void XValInstanceStream::startSubstream(const unsigned int fold, const bool training, const bool building) {
    fold_ = fold;
    training_ = training;
    building_ = building;
    cou = 0;
    rewind();
}

void XValInstanceStream::setflag(int flag) {
    flag_ = flag;
}

void XValInstanceStream::setdivide(unsigned int d) {
    divide = d;
}
/// return to the first instance in the stream

void XValInstanceStream::rewind() {
    source_->rewind();
    rand_.seed(seed_);
    count_ = 0;
}

/// advance, discarding the next instance in the stream.  Return true iff successful.

bool XValInstanceStream::advance() {
    if (flag_ == 0) {
        while (source_->advance()) {
            int a = rand_();
            if (a % noOfFolds_ == fold_) {
                //   printf("rand: %u no: %u\n",a,noOfFolds_);
                // test instance
                if (!training_) {
                    count_++;
                    return true;
                }
            } else {
                // training instance
                if (training_) {
                    count_++;
                    return true;
                }
            }
        }
    } else {
        while (source_->advance()) {
            unsigned long a = rand_();
            if (a % noOfFolds_ == fold_) {
                printf("rand: %u no: %u\n", a, noOfFolds_);
                // test instance
                if (!training_) {
                    count_++;
                    return true;
                }
            } else {
                // training instance
                if (training_) {
                    int b = (fold_ + divide) % 5;
                    if (a % 5 == b) {

                        if (!building_) {
                            count_++;
                            return true;
                        }
                    } else {
                        if (building_)
                            count_++;
                        return true;
                    }
                }
            }
        }
    }
    return false;
}
/*bool XValInstanceStream::advance(instance &inst) {
  while (!source_->isAtEnd()) {
    if (rand_() % noOfFolds_ == fold_) {
      // test instance
      if (training_) {
        source_->advance();
      }
      else {
        // testing
        if (source_->advance(inst)) {
          count_++;
          return true;
        }
        else return false;
      }
    }
    else {
      // training instance
      if (training_) {
        if (source_->advance(inst)) {
          count_++;
          return true;
        }
        else return false;
      }
      else {
        // testing
        source_->advance();
      }
    }
  }
  return false;
}*/
/// advance to the next instance in the stream.Return true iff successful. @param inst the instance record to receive the new instance. 

bool XValInstanceStream::advance(instance &inst) {

    while (!source_->isAtEnd()) {
        unsigned long ra = rand_();
        cou++;
        // cout<<"rand"<<rand_()<<" "<<ra<<endl;
        //printf("rand: %  no: %d  r: %d\n",ra,noOfFolds_,ra% noOfFolds_);
        if (ra % noOfFolds_ == fold_) {

            // printf("test\n");
            if (training_) {
                source_->advance();
            } else {
                // testing
                if (source_->advance(inst)) {
                    count_++;
                    return true;
                } else return false;
            }
        } else {
            // training instance
            //  printf("train\n");
            if (flag_ == 0) {
                // printf("train flag:%u \n",flag_);
                if (training_) {
                    if (source_->advance(inst)) {
                        count_++;
                        return true;
                    } else return false;
                } else {
                    // testing
                    source_->advance();
                }
            } else {
                if (training_) {
                    int a = (fold_ + divide + 1) % 5;
                    //            if(cou<20)
                    //            printf(" ra :%d a aa:%d mod:%d\n",ra,a,(ra%5));
                    // if(a==fold_)a++;
                    if ((ra % 5) == a) {
                        //  printf("yu yuyu building: %d\n",building_);
                        if (building_) {
                            source_->advance();
                            // printf("train and build: \n");
                        } else {
                            // testing
                            if (source_->advance(inst)) {
                                count_++;
                                //                             if(cou<20)
                                //                             printf("count；%u \n",ra);
                                //  printf("BB train and build: \n");
                                return true;
                            } else return false;
                        }
                    } else {
                        if (building_) {
                            if (source_->advance(inst)) {
                                count_++;
                                //   printf("cc train and build: \n");
                                return true;
                            } else return false;
                        } else {
                            // testing
                            source_->advance();
                        }
                    }

                } else {
                    // testing
                    source_->advance();
                    // printf("test\n");
                }
            }
        }
    }

    return false;
}

/// true if we have advanced past the last instance

bool XValInstanceStream::isAtEnd() {
    return source_->isAtEnd();
}

/// the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.  The stream state is undefined after a call to size(), so a rewind shouldbe performed before the next advance.

InstanceCount XValInstanceStream::size() {
    if (!isAtEnd()) {
        instance inst(*this);

        while (!isAtEnd()) advance(inst);
    }

    return count_;
}
