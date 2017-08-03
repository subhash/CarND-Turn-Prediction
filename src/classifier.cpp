#include <iostream>
#include <sstream>
#include <fstream>
#include <numeric>
#include <math.h>
#include <vector>
#include "classifier.h"

using namespace std;

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}


vector<double> extract_features(vector<double> data) {
  double s = data[0], d = data[1], s_dot = data[2], d_dot = data[3];
  return {  s, fmod(d, 4.0), s_dot, d_dot };
}


double average(vector<double> v) {
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  return sum / v.size();
}

double stddev(vector<double> v, double mean) {
  double diff = std::accumulate(v.begin(), v.end(), 0.0, [mean](double s, double x) { return s + (x-mean) * (x-mean); } );
  return sqrt( diff / v.size());
}

vector<vector<vector<double>>> GNB::train(vector<vector<double>> data, vector<string> labels)
{

  /*
    Trains the classifier with N data points and labels.

    INPUTS
    data - array of N observations
      - Each observation is a tuple with 4 values: s, d,
        s_dot and d_dot.
      - Example : [
          [3.5, 0.1, 5.9, -0.02],
          [8.0, -0.3, 3.0, 2.2],
          ...
        ]

    labels - array of N labels
      - Each label is one of "left", "keep", or "right".
  */

  //vector<double> lf[3][4];
  vector<vector<vector<double>>> lf(n_labels, vector<vector<double>>(n_features));


  for (int i=0; i<data.size(); i++) {
    int label_index = find(possible_labels.begin(), possible_labels.end(), labels[i]) - possible_labels.begin();

    auto features = extract_features(data[i]);
    for (int f=0; f<n_features; f++) {
      lf[label_index][f].push_back(features[f]);
    }
  }

  for (int l=0; l<lf.size(); l++) {
    for (int f=0; f<lf[l].size(); f++) {
      auto& feature = lf[l][f];
      double mean = average(feature);
      double sd = stddev(feature, mean);
      stat[l][f].push_back(mean);
      stat[l][f].push_back(sd);
    }
  }


  return lf;

}

std::pair<string, double> GNB::predict(vector<double> sample)
{
  /*
    Once trained, this method is called and expected to return
    a predicted behavior for the given observation.

    INPUTS

    observation - a 4 tuple with s, d, s_dot, d_dot.
      - Example: [3.5, 0.1, 8.5, -0.2]

    OUTPUT

    A label representing the best guess of the classifier. Can
    be one of "left", "keep" or "right".
    """
    # TODO - complete this
  */
  vector<double> features = extract_features(sample);
  vector<double> probs;
  for(int l=0; l<n_labels; l++) {
    double pdf = 1.0;
    for(int f=0; f<n_features; f++) {
      double mu = stat[l][f][0], sigma = stat[l][f][1];
      pdf *= (1/sqrt(2*M_PI*sigma*sigma))*exp(-(pow(features[f]-mu, 2)/(2*sigma*sigma)));
    }
    probs.push_back(pdf);
  }
  int mx = max_element(probs.begin(), probs.end()) - probs.begin();
  double total_prob = std::accumulate(probs.begin(), probs.end(), 0.0);

  return std::make_pair(this->possible_labels[mx], probs[mx]/total_prob);

}
