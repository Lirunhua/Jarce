/*
 * extLearnLibSVM.h
 *
 *  Created on: 14/09/2012
 *      Author: nayyar
 */
#pragma once
#include "externLearner.h"
#include "instanceStream.h"

#include "stdlib.h"
#include "utils.h"

class extLearnLibSVM : public externLearner {
private:
  FILE *testf;
  FILE *trainf;
  char*const* args;  // arguments to pass to executable
  int argcnt; // count of args
  std::string tmpTrain;
  std::string tmpTest;

public:
  extLearnLibSVM(char *wd, char *bd, char*const argv[], int argc) : externLearner(wd, bd), args(argv), argcnt(argc)
  {
    tmpTrain = workingDir;
    tmpTrain += "/tmp.train";
    tmpTest = workingDir;
    tmpTest += "/tmp.test";
  }
  ~extLearnLibSVM() { }

  void printTrainingData(InstanceStream *instanceStream);
  void printMeta(InstanceStream *instanceStream) const;
  void classify(InstanceStream *instanceStream, int fold);
  void printResults(InstanceStream *instanceStream);

private:
  void printData(const char* filename, InstanceStream *instanceStream);
};

