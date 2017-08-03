#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <algorithm>

using namespace std;

class GNB {

private:



public:

  vector<string> possible_labels = {"left","keep","right"};



  /**
    * Constructor
    */
  GNB();

  /**
  * Destructor
  */
  virtual ~GNB();

  vector<vector<vector<double>>> train(vector<vector<double> > data, vector<string>  labels);

  std::pair<string, double> predict(vector<double>);


private:
    const int n_labels = possible_labels.size();
    const int n_features = 4;
    vector<vector<vector<double>>> stat = vector<vector<vector<double>>>(n_labels, vector<vector<double>>(n_features));


};

#endif



