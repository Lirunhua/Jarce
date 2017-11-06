/* Petal: An open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
**
** Module for registering each of the available learners.
** WOuld ideally use a map from char* to learner constructors, but c++ does not seem to support that
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
#include "learnerRegistry.h"
#include "utils.h"

// learners
#include "baggedLearner.h"
#include "DTree.h"
#include "ensembleLearner.h"
#include "featingLearner.h"
#include "kdb.h"
#include "entropykdb.h"
#include "cmkdb.h"
#include "localkdb.h"
#include "cmikdb.h"
#include "trainclasskdb.h"
#include "decisiontreekdb.h"
//**************************************
#include "dpkdb_alldel.h"
#include "dpkdb_ndp_k2.h"
#include "dpkdb_recursion_del.h"
#include "dpkdb_recursion_k2.h"
#include "dpkdbrec.h"
#include "sortkdb.h"//<---------------------
#include "sortkdb_del.h"

#include "kdb_PH_norder.h"
#include "kdb_minH.h"
#include "dpkdb_rec_p.h"
#include "testan.h"
#include "test_cmi.h"

//**************************************
//**************************************
#include "kdbtest.h"
#include "tantest.h"


//**************************************
#include "kdbEager.h"
#include "vfan.h"

#include "hnb.h"
#include "myhnb.h"
#include "myhnb_new.h"
#include "hnb_aode.h"
#include "waode.h"
#include "hnb_waode.h"

#include "kdbExt.h"
#include "kdbCondDisc.h"
#include "kdbCondDisc2.h"
#include "nb.h"
#include "tan_gen.h"
#include "TAN_pro.h"
#include "tan.h"
#include "tctan.h"
#include "random.h"
#include "RFDTree.h"
#include "aodeDist.h"
#include "aode.h"
#include "aodeChen.h"
#include "aodeEager.h"
#include "a2de.h"
#include "a2de2.h"
#include "a2de3.h"
#include "a3de.h"
#include "aodeselect.h"
#include "sample.h"
//#include "kdbext2.h"
#include "aode_order_ConMur.h"
#include "aodedouble.h"
#include "aodeMarkov.h"
#include "kdbOri.h"
#include "kdbext2.h"

// create a new learner, selected by name
learner *createLearner(const char *leanername, char*const*& argv, char*const* end) {
  if (streq(leanername, "aode")) {
    return new aode(argv, end);
  }
  if (streq(leanername, "sample")) {
    return new sample(argv, end);
  }
  if (streq(leanername, "aode-eager")) {
    return new aodeEager(argv, end);
  }
   if (streq(leanername, "aodeselect")) {
    return new aodeselect(argv, end);
  }
  else if (streq(leanername, "aodeDist")) {
	    return new aodeDist(argv, end);
	  }
  else if (streq(leanername, "a2de")) {
    return new a2de(argv, end);
  }
  else if (streq(leanername, "a2de2")) {
    return new a2de2(argv, end);
  }
  else if (streq(leanername, "a2de3")) {
    return new a2de3(argv, end);
  }
  else if (streq(leanername, "a3de")) {
    return new a3de(argv, end);
  }
  else if (streq(leanername, "bagging")) {
    return new BaggedLearner(argv, end);
  }
  else if (streq(leanername, "dtree")) {
    return new DTree(argv, end);
  }
  else if (streq(leanername, "ensembled")) {
    return new EnsembleLearner(argv, end);
  }
  else if (streq(leanername, "feating")) {
    return new FeatingLearner(argv, end);
  }
  else if (streq(leanername, "kdb")) {
    return new kdb(argv, end);
  }
  else if (streq(leanername, "entropykdb")) {
    return new entropykdb(argv, end);
  }
  else if (streq(leanername, "cmkdb")) {
    return new cmkdb(argv, end);
  }
  else if (streq(leanername, "localkdb")) {
    return new localkdb(argv, end);
  }
  else if (streq(leanername, "cmikdb")) {
    return new cmikdb(argv, end);
  }
  else if (streq(leanername, "trainclasskdb")) {
    return new trainclasskdb(argv, end);
  }
  else if (streq(leanername, "decisiontreekdb")) {
    return new decisiontreekdb(argv, end);
  }
  //****************************************
  else if (streq(leanername, "dpkdb_alldel")) {
    return new dpkdb_alldel(argv, end);
  }
  else if (streq(leanername, "dpkdb_ndp_k2")) {
    return new dpkdb_ndp_k2(argv, end);
  }
  else if (streq(leanername, "dpkdb_recursion_del")) {
    return new dpkdb_recursion_del(argv, end);
  }
  else if (streq(leanername, "dpkdb_recursion_k2")) {
    return new dpkdb_recursion_k2(argv, end);
  }
  else if (streq(leanername, "kdb_PH_norder")) {
    return new kdb_PH_norder(argv, end);
  }
  else if (streq(leanername, "kdb_minH")) {
    return new kdb_minH(argv, end);
  }
  else if (streq(leanername, "dpkdb_rec_p")) {
    return new dpkdb_rec_p(argv, end);
  }
  else if (streq(leanername, "testan")) {
    return new testan(argv, end);
  }  
  else if (streq(leanername, "test_cmi")) {
    return new test_cmi(argv, end);
  }  
  else if (streq(leanername, "dpkdbrec")) {
    return new dpkdbrec(argv, end);
  }
  else if (streq(leanername, "sortkdb")) {
    return new sortkdb(argv, end);
  }
  else if (streq(leanername, "sortkdb_del")) {
    return new sortkdb_del(argv, end);
  }
  
  
 //********************************************************
  else if (streq(leanername, "kdbtest")) {
    return new kdbtest(argv, end);
  }
  else if (streq(leanername, "tantest")) {
    return new tantest(argv, end);
  }
  
  
  
  
  
  
  
  
  
  //*******************************************************
  //****************************************
  else if (streq(leanername, "kdb-eager")) {
    return new kdbEager(argv, end);
  }
  //else if (streq(learnername, "kdb2")) {
  //  return new kdb2(argv, end);
  //}
  if(streq(leanername,"hnb")){
      return new hnb(argv,end);
  }
  if(streq(leanername,"myhnb")){
      return new myhnb(argv,end);
  }
  if(streq(leanername,"myhnb_new")){
      return new myhnb_new(argv,end);
  }
  if(streq(leanername,"hnb_aode")){
      return new hnb_aode(argv,end);
  }
  if(streq(leanername,"waode")){
      return new waode(argv,end);
  }
  if(streq(leanername,"hnb_waode")){
      return new hnb_waode(argv,end);
  }
  
 if (streq(leanername, "vfan")) {
    return new vfan(argv, end);
  }
  else if (streq(leanername, "kdb-condDisc")) {
    return new kdbCondDisc(argv, end);
  }
  else if (streq(leanername, "kdb-condDisc2")) {
    return new kdbCondDisc2(argv, end);
  }
  else if (streq(leanername, "nb")) {
    return new nb(argv, end);
  }
  else if (streq(leanername, "random")) {
    return new randomClassifier(argv, end);
  }
  else if (streq(leanername, "rfdtree")) {
    return new RFDTree(argv, end);
  }
  else if (streq(leanername, "tan")) {
    return new TAN(argv, end);
  }
  else if (streq(leanername, "tctan")) {
    return new tctan(argv, end);
  }
  else if (streq(leanername, "tan_gen")) {
    return new TAN_gen(argv, end);
  }
   else if (streq(leanername, "TAN_pro")) {
    return new TAN_pro(argv, end);
  }
   else if (streq(leanername, "aode_order_ConMur")) {
    return new aode_order_ConMur(argv, end);
  }
   else if (streq(leanername, "aodeDouble")) {
    return new aodeDouble(argv, end);
  }
   else if (streq(leanername, "aodeMarkov")) {
    return new aodeMarkov(argv, end);  }
  
   else if (streq(leanername, "kdbOri")) {
    return new kdbOri(argv, end);
  }
  else if (streq(leanername, "aodeChen")) {
    return new aodeChen(argv, end);
  }

  return NULL;
}

