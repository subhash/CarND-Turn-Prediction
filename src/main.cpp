#include "classifier.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using namespace std;

vector<vector<double> > Load_State(string file_name)
{
    ifstream in_state_(file_name.c_str(), ifstream::in);
    vector< vector<double >> state_out;
    string line;


    while (getline(in_state_, line))
    {
        istringstream iss(line);
      vector<double> x_coord;

      string token;
      while( getline(iss,token,','))
      {
          x_coord.push_back(stod(token));
      }
      state_out.push_back(x_coord);
    }
    return state_out;
}
vector<string> Load_Label(string file_name)
{
    ifstream in_label_(file_name.c_str(), ifstream::in);
    vector< string > label_out;
    string line;
    while (getline(in_label_, line))
    {
      istringstream iss(line);
      string label;
      iss >> label;

      label_out.push_back(label);
    }
    return label_out;

}

int main() {

    vector< vector<double> > X_train = Load_State("./train_states.txt");
    vector< vector<double> > X_test  = Load_State("./test_states.txt");
    vector< string > Y_train  = Load_Label("./train_labels.txt");
    vector< string > Y_test   = Load_Label("./test_labels.txt");

  cout << "X_train number of elements " << X_train.size() << endl;
  cout << "X_train element size " << X_train[0].size() << endl;
  cout << "Y_train number of elements " << Y_train.size() << endl;

  GNB gnb = GNB();

  vector<vector<vector<double>>> lf = gnb.train(X_train, Y_train);

  cout << "X_test number of elements " << X_test.size() << endl;
  cout << "X_test element size " << X_test[0].size() << endl;
  cout << "Y_test number of elements " << Y_test.size() << endl;

  int score = 0;
  for(int i = 0; i < X_test.size(); i++)
  {
    vector<double> coords = X_test[i];
    std::pair<string, double> predicted = gnb.predict(coords);
    string prediction = predicted.first;
    double confidence = predicted.second;
    bool correct = (prediction.compare(Y_test[i]) == 0);
    if(correct)
    {
      score += 1;
    } else std::cout  << " " << prediction << " <> " << Y_test[i] << ", " << confidence << " - " << coords[0] << "," <<  coords[1 ]<< "," <<  coords[2]<< "," << coords[3]<< "," <<std::endl;
  }

  float fraction_correct = float(score) / Y_test.size();
  cout << "You got " << (100*fraction_correct) << " correct" << endl;


  plt::subplot(2,2,1);
  plt::title("s");
  plt::plot(lf[0][0], "ro");
  plt::plot(lf[1][0], "bo");
  plt::plot(lf[2][0], "gx");

  plt::subplot(2,2,2);
  plt::title("d");
  plt::plot(lf[0][1], "ro");
  plt::plot(lf[1][1], "bo");
  plt::plot(lf[2][1], "gx");


  plt::subplot(2,2,3);
  plt::title("s dot");
  plt::plot(lf[0][2], "ro");
  plt::plot(lf[1][2], "bo");
  plt::plot(lf[2][2], "gx");

  plt::subplot(2,2,4);
  plt::title("d dot");

  plt::plot(lf[0][3], "ro");
  plt::plot(lf[1][3], "bo");
  plt::plot(lf[2][3], "gx");


  plt::show();

  return 0;
}


