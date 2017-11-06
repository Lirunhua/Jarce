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
#include "xxyDistEager.h"
#include "utils.h"
#include <assert.h>

xxyDistEager::xxyDistEager() {
}

xxyDistEager::xxyDistEager(InstanceStream& stream) {
	reset(stream);

	// pass through the stream updating the counts incrementally
	stream.rewind();

	instance i;

	while (stream.advance(i)) {
		update(i);
	}
}

xxyDistEager::~xxyDistEager(void) {
}

void xxyDistEager::reset(InstanceStream& stream) {
	instanceStream_ = &stream;

	xyCounts.reset(&stream);

	noCatAtts_ = instanceStream_->getNoCatAtts();
	noClasses_ = instanceStream_->getNoClasses();

	//initialise the xxyCounts
	count.resize(noCatAtts_);
	for (CategoricalAttribute i = 1; i < noCatAtts_; i++) {

		CatValue firstNoValues = instanceStream_->getNoValues(i);

		count[i].resize(firstNoValues);

		for (CatValue j = 0; j < firstNoValues; j++) {
			count[i][j].resize(i);

			for (CategoricalAttribute k = 0; k < i; k++) {

				CatValue secondNoValues = instanceStream_->getNoValues(k);

				count[i][j][k].resize(secondNoValues);

				for (CatValue l = 0; l < secondNoValues; l++) {

					count[i][j][k][l].assign(noClasses_,0);
				}
			}
		}
	}

	//initialise the conditional probability
	condiProbs.resize(noCatAtts_);
	for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {

		CatValue firstNoValues = instanceStream_->getNoValues(i);

		condiProbs[i].resize(firstNoValues);

		for (CatValue j = 0; j < firstNoValues; j++) {
			condiProbs[i][j].resize(noCatAtts_);

			for (CategoricalAttribute k = 0; k < noCatAtts_; k++) {

				CatValue secondNoValues = instanceStream_->getNoValues(k);

				condiProbs[i][j][k].resize(secondNoValues);

				for (CatValue l = 0; l < secondNoValues; l++) {

					condiProbs[i][j][k][l].assign(noClasses_, 1.0);
				}
			}
		}
	}

}

void xxyDistEager::update(const instance& i) {
	xyCounts.update(i);

	CatValue theClass = i.getClass();

	for (CategoricalAttribute x1 = 1; x1 < instanceStream_->getNoCatAtts();
			x1++) {
		CatValue v1 = i.getCatVal(x1);

		std::vector<std::vector<std::vector<InstanceCount> > > *countsForAtt =
				&count[x1][v1];

		for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

			CatValue v2 = i.getCatVal(x2);

			(*countsForAtt)[x2][v2][theClass]++;

			//	assert(count[x1][v1][x2][v2 * noClasses_ + theClass] <= count);
		}
	}
}
void xxyDistEager::calculateCondProb() {



	//compute the conditional probability

	for (CategoricalAttribute i = 1; i < noCatAtts_; i++) {
		CatValue firstNoValues = instanceStream_->getNoValues(i);

		for (CatValue j = 0; j < firstNoValues; j++) {

			std::vector<std::vector<std::vector<InstanceCount> > > *parentVecotr =
					&count[i][j];

			for (CategoricalAttribute k = 0; k < i; k++) {
				CatValue secondNoValues = instanceStream_->getNoValues(k);
				for (CatValue l = 0; l < secondNoValues; l++) {

					std::vector<InstanceCount> *childVector2 =
							&(*parentVecotr)[k][l];

					for (CatValue y = 0; y < noClasses_; y++) {

						double parentYM = xyCounts.getCount(i, j, y) + M;

						double childYM = xyCounts.getCount(k, l, y) + M;

						InstanceCount parentChildY = (*childVector2)[y];

						condiProbs[i][j][k][l][y] = (parentChildY
								+ M / firstNoValues) / childYM;
						condiProbs[k][l][i][j][y] = (parentChildY
								+ M / secondNoValues) / parentYM;

					}

				}
			}
		}
	}

}
