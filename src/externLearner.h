/*
 * extLearn.h
 *
 *  Created on: 14/09/2012
 *      Author: nayyar
 */
#pragma once
#include "instanceStream.h"

class externLearner {
public:
	externLearner(char *wd, char *bd) : workingDir(wd), binDir(bd) {}

	virtual void printTrainingData(InstanceStream *instanceStream) = 0;
	virtual void printMeta(InstanceStream *instanceStream) const = 0;
	virtual void classify(InstanceStream *instanceStream, int fold) = 0;
        virtual void printResults(InstanceStream *instanceStream) = 0;

protected:
	char *workingDir; // the working directory
	char *binDir; // the directory with the binaries to execute
};

