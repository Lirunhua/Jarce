/*
 * extLearnPetal.cpp
 *
 *  Created on: 14/09/2012
 *      Author: nayyar
 */

#include "extLearnPetal.h"

void extLearnPetal::printTrainingData(InstanceStream *instanceStream) {
  printData("tmp.train", instanceStream);
}

void extLearnPetal::printData(const char* filename, InstanceStream *instanceStream) {
  FILE* f = fopen(filename, "w");
  if (f == NULL) error("Cannot open output file %s", filename);
  
  instance inst(*instanceStream);

  while (instanceStream->advance(inst)) {
    for (CategoricalAttribute a = 0; a < instanceStream->getNoCatAtts(); a++) {
      fprintf(f, "%s,", instanceStream->getCatAttValName(a, inst.getCatVal(a)));
    }
    for (NumericAttribute a = 0; a < instanceStream->getNoNumAtts(); a++) {
      fprintf(f, "%0.*f,", instanceStream->getPrecision(a), inst.getNumVal(a));
    }
    fputs(instanceStream->getClassName(inst.getClass()), f);
    fputc('\n',f);
  }

  fclose(f);
}

void extLearnPetal::printMeta(InstanceStream *instanceStream) const {
  instanceStream->printMetadata("tmp.meta");
}

void extLearnPetal::classify(InstanceStream *instanceStream, int fold) {
  printData("tmp.test",  instanceStream);
  system("./petal tmp.meta tmp.train -ttmp.test");
  remove("tmp.meta");
  remove("tmp.train");
  remove("tmp.test");
}

void extLearnPetal::printResults(InstanceStream *instanceStream){
  
}



