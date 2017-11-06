#include "xxxyDist3.h"
#include "xxxyDist.h"
#include "globals.h"
#include "utils.h"
xxxyDist3::xxxyDist3() {
}

xxxyDist3::xxxyDist3(InstanceStream& stream) {
	reset(stream);

	// pass through the stream updating the counts incrementally
	stream.rewind();

	instance i;

	while (stream.advance(i)) {
		update(i);
	}

}

void xxxyDist3::setOrder(std::vector<CategoricalAttribute> &order) {
	order_ = order;
}
void xxxyDist3::setNoSelectedCatAtts(unsigned int noSelectedCatAtts) {
	noSelectedCatAtts_ = noSelectedCatAtts;
}

void xxxyDist3::reset(InstanceStream& stream) {

	metaData_ = stream.getMetaData();

	xxyCounts.reset(stream);

	noCatAtts_ = metaData_->getNoCatAtts();
	noClasses_ = metaData_->getNoClasses();

	noUnSelectedCatAtts_ = noCatAtts_ - noSelectedCatAtts_;

	//set space for xxxyCount
	xxxyCount.resize(noSelectedCatAtts_);

	//set space for xxXyCount
	xxXyCount.resize(noSelectedCatAtts_);

	for (CategoricalAttribute x1 = 1; x1 < noSelectedCatAtts_; x1++) {

		//second vector
		xxxyCount[x1].resize(getNoValues(x1) * x1);

		xxXyCount[x1].resize(getNoValues(x1) * x1);

		for (CatValue v1 = 0; v1 < getNoValues(x1); v1++) {
			for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

				//third vector
				xxxyCount[x1][v1 * x1 + x2].resize(getNoValues(x2) * x2);
				xxXyCount[x1][v1 * x1 + x2].resize(
						getNoValues(x2) * noUnSelectedCatAtts_);

				for (CatValue v2 = 0; v2 < getNoValues(x2); v2++) {

					//allocate for the selected attribute
					for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {

						//inner vector
						xxxyCount[x1][v1 * x1 + x2][v2 * x2 + x3].assign(
								getNoValues(x3) * noClasses_, 0);
					}
					//allocate for the unselected attributes
					for (CategoricalAttribute x3 = 0; x3 < noUnSelectedCatAtts_;
							x3++) {

						//inner vector
						xxXyCount[x1][v1 * x1 + x2][v2 * noUnSelectedCatAtts_
								+ x3].assign(getNoValues(x3+noSelectedCatAtts_) * noClasses_, 0);
					}
				}
			}
		}
	}

}

xxxyDist3::~xxxyDist3(void) {

}

void xxxyDist3::update(const instance &i) {
	xxyCounts.update(i);

	CatValue theClass = i.getClass();

	for (CategoricalAttribute x1 = 1; x1 < noSelectedCatAtts_; x1++) {
		CatValue v1 = i.getCatVal(order_[x1]);

		XXYSubDist xxySubDist(getXXYSubDist(x1, v1), noClasses_);

		XXYSubDist xxySubDistRest(getXXYSubDistRest(x1, v1), noClasses_);

		for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
			CatValue v2 = i.getCatVal(order_[x2]);

			XYSubDist xySubDist(xxySubDist.getXYSubDist(x2, v2), noClasses_);

			XYSubDist xySubDistRest(
					xxySubDistRest.getXYSubDist(x2, v2, noUnSelectedCatAtts_),
					noClasses_);

			for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {
				CatValue v3 = i.getCatVal(order_[x3]);

				xySubDist.incCount(x3, v3, theClass);
				//xxySubDist.incCount(x2, v2, x3, v3, theClass);

				assert(
						*ref(x1,v1,x2,v2,x3,v3,theClass) <= xxyCounts.xyCounts.count);
			}

			for (CategoricalAttribute x3 = 0; x3 < noUnSelectedCatAtts_; x3++) {
				CatValue v3 = i.getCatVal(order_[x3 + noSelectedCatAtts_]);

//				xySubDist.incCount(x3,v3,theClass);
				xySubDistRest.incCount(x3, v3, theClass);

				assert(
						*refRest(x1,v1,x2,v2,x3,v3,theClass) <= xxyCounts.xyCounts.count);
			}
		}
	}

}

void xxxyDist3::clear() {
	xxxyCount.clear();
	xxXyCount.clear();
	xxyCounts.clear();
}
