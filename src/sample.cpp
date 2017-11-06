/* Open source system for classification learning from very large data
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

#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>

#include "utils.h"
#include "sample.h"

sample::sample(char* const *& argv, char* const * end) :
		trainingIsFinished_(false) {
	name_ = "Sample";
	count = 0;
	Cnt_ = 0; //unless specified it is going to sample/discretise all instances
	while (argv != end) {
		if (*argv[0] != '-') {
			break;
		} else if (argv[0][1] == 'n') {
			getUIntFromStr(argv[0] + 2, Cnt_, "n");
		} else if (argv[0][1] == 'o') {
			filename_ = &argv[0][2];
		} 
		++argv;
	}

	trainingIsFinished_ = false;
        
}

sample::~sample(void) {
}

void sample::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void sample::reset(InstanceStream &is) {

	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();
	trainingIsFinished_ = false;
}

void sample::train(const instance &inst) {

	count++;
	if ( (Cnt_!=0) && (count > Cnt_) )
		return;
	for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                 fprintf(f, "%d,", inst.getCatVal(a));
		}
        fprintf(f, "%d\n", inst.getClass());
}

void sample::initialisePass() {
	f = fopen(filename_, "w");
}

void sample::finalisePass() {
	fclose(f);
	trainingIsFinished_ = true;
}

bool sample::trainingIsFinished() {

	return trainingIsFinished_;
}

void sample::classify(const instance &inst, std::vector<double> &classDist) {

}

