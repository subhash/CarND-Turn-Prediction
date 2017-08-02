#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

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

  void train(vector<vector<double> > data, vector<string>  labels);

    string predict(vector<double>);


private:
    const int n_labels = possible_labels.size();
    const int n_features = 4;
    vector<vector<vector<double>>> stat = vector<vector<vector<double>>>(n_labels, vector<vector<double>>(n_features));

    //vector<double> extract_features(vector<double> data);

};

#endif



