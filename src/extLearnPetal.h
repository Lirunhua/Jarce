/*
 * extLearnPetal.h
 *
 *  Created on: 14/09/2012
 *      Author: nayyar
 */
#pragma once
#include "externLearner.h"
#include "instance.h"

#include "stdlib.h"
#include "utils.h"

class extLearnPetal : public externLearner {
public:
  extLearnPetal(char *wd, char *bd) : externLearner(wd, bd) {}
  ~extLearnPetal() { }

  void printTrainingData(InstanceStream *instanceStream);
  void printMeta(InstanceStream *instanceStream) const;
  void classify(InstanceStream *instanceStream, int fold);
  void printResults(InstanceStream *instanceStream);

private:
  void printData(const char* filename, InstanceStream *instanceStream);
};
