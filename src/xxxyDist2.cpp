#include "xxxyDist2.h"
#include "xxxyDist.h"
#include "globals.h"
#include "utils.h"
xxxyDist2::xxxyDist2() {
}

xxxyDist2::xxxyDist2(InstanceStream& stream) {
	reset(stream);

	// pass through the stream updating the counts incrementally
	stream.rewind();

	instance i;

	while (stream.advance(i)) {
		update(i);
	}

}

void xxxyDist2::setOrder(std::vector<CategoricalAttribute> &order) {
	order_ = order;
}
void xxxyDist2::setNoSelectedCatAtts(unsigned int noSelectedCatAtts)
{
	noSelectedCatAtts_=noSelectedCatAtts;
}

void xxxyDist2::reset(InstanceStream& stream) {

	metaData_ = stream.getMetaData();

	xxyCounts.reset(stream);

	noCatAtts_ = metaData_->getNoCatAtts();
	noClasses_ = metaData_->getNoClasses();



	//out vector
	count.resize(noSelectedCatAtts_);
	for (CategoricalAttribute x1 = 1; x1 < noSelectedCatAtts_; x1++) {

		//second vector
		count[x1].resize(getNoValues(x1) * x1);

		for (CatValue v1 = 0; v1 < getNoValues(x1); v1++) {
			for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

				//third vector
				count[x1][v1 * x1 + x2].resize(getNoValues(x2) * noCatAtts_);
				for (CatValue v2 = 0; v2 < getNoValues(x2); v2++) {
					for (CategoricalAttribute x3 = 0; x3 < noCatAtts_; x3++) {

						//inner vector
						count[x1][v1 * x1 + x2][v2 * noCatAtts_ + x3].assign(
								getNoValues(x3) * noClasses_, 0);
					}
				}
			}
		}
	}
}

xxxyDist2::~xxxyDist2(void) {

}

void xxxyDist2::update(const instance &i) {
	xxyCounts.update(i);

	CatValue theClass = i.getClass();

	for (CategoricalAttribute x1 = 1; x1 < noSelectedCatAtts_; x1++) {
		CatValue v1 = i.getCatVal(order_[x1]);

		XXYSubDist2 xxySubDist(getXXYSubDist(x1, v1), noClasses_,noCatAtts_);

		for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
			CatValue v2 = i.getCatVal(order_[x2]);

//		    XYSubDist xySubDist(getXYSubDist(x1, v1,x2,v2), noClasses_);

			for (CategoricalAttribute x3 = 0; x3 < noCatAtts_; x3++) {
				CatValue v3 = i.getCatVal(order_[x3]);

//				xySubDist.incCount(x3,v3,theClass);
				xxySubDist.incCount(x2, v2, x3, v3, theClass);

//				if(!(*ref(x1,v1,x2,v2,x3,v3,theClass) <= xxyCounts.xyCounts.count))
//					printf("error!\n");

				assert(
						*ref(x1,v1,x2,v2,x3,v3,theClass) <= xxyCounts.xyCounts.count);
			}
		}
	}

}

void xxxyDist2::clear() {
	count.clear();
	xxyCounts.clear();
}
