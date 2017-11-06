/*
 * extLearnSVMSGD.cpp
 *
 *  Created on: 17/09/2012
 *      Author: nayyar
 */

#include "extLearnSVMSGD.h"

void extLearnSVMSGD::printTrainingData(InstanceStream *instanceStream) {
  printData(tmpTrain.c_str(), instanceStream);
}

void extLearnSVMSGD::printData(const char* filename, InstanceStream *instanceStream) {
  FILE* f = fopen(filename, "w");
  if (f == NULL) error("Cannot open output file %s", filename);
  
  instance inst(*instanceStream);

  while (instanceStream->advance(inst)) {
    if (instanceStream->getNoClasses() == 2) {
      if (inst.getClass() == 1) {
        fprintf(f,"%d",1);
      } else {
        fprintf(f,"%d",-1);
      }
    } else
      fprintf(f,"%d",inst.getClass());

    for (CategoricalAttribute a = 0; a < instanceStream->getNoCatAtts(); a++) {
      error("Can't handle categorical values");
    }
    for (NumericAttribute a = 0; a < instanceStream->getNoNumAtts(); a++) {
      if (inst.getNumVal(a) == 0 || inst.isMissing(a)) {
      } else {
        fprintf(f, " %d:%0.*f", a+1, instanceStream->getPrecision(a), inst.getNumVal(a));
      }
    }

    fputc('\n',f);
  }

  fclose(f);
}

void extLearnSVMSGD::printMeta(InstanceStream *instanceStream) const {
  // does not use a meta file
}

void extLearnSVMSGD::classify(InstanceStream *instanceStream, int fold) {
  printData(tmpTest.c_str(),  instanceStream);

  // the temporary files
  std::string tmpModel = workingDir;
  tmpModel += "/tmp.model";
  std::string tmpOutput = workingDir;
  tmpOutput += "/tmp.output";

  std::string svmTrain;

  svmTrain = binDir;
  svmTrain += "/svmsgdHL";
  for (int i = 0; i < argcnt; i++) {
    svmTrain += ' ';
    svmTrain += args[i];
  }
  svmTrain += ' '; svmTrain += tmpTrain.c_str();
  svmTrain += ' '; svmTrain += tmpTest.c_str();

  printf("Executing commnad - %s\n", svmTrain.c_str());
  printf("Calling SVMSGD\n"); system(svmTrain.c_str()); printf("Finished calling SVMSGD\n");

  //remove("tmp.meta");
  printf("Deleting files \n");
  remove(tmpTrain.c_str());
  remove(tmpTest.c_str());
  printf("Finished deleting files\n");
}

void extLearnSVMSGD::printResults(InstanceStream *instanceStream){
  
}



