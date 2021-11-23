#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if (estimations.size() == 0){
      std::cout << "Tools::CalculateRMSE: Estimations vector is empty!" << std::endl;
      return rmse;
  } else if (estimations.size() != ground_truth.size()) {
      std::cout << "Tools::CalculateRMSE: Estimations and ground truth vectors must have the same size!" << std::endl;
      return rmse;
  }

  for (int i=0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  rmse /= estimations.size();
  rmse = rmse.array().sqrt();
  
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Jacobian(3,4);
  Jacobian << 0,0,0,0,
              0,0,0,0,
              0,0,0,0;
  
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  
  float p_distance = sqrt(pow(px, 2) + pow(py, 2));
  float p_distance_cubed = pow(p_distance, 3);
  float vx_py = vx*py;
  float vy_px = vy*px;

  // check division by zero
  if (p_distance < 0.000001) {
      std::cout << "Tools::CalculateJacobian(): ERROR -- Division by zero!" << std::endl;
    return Jacobian;
  } 
  
  // compute the Jacobian matrix
  Jacobian(0, 0) = px / p_distance;
  Jacobian(0, 1) = py / p_distance;
  Jacobian(1, 0) = -py / p_distance;
  Jacobian(1, 1) = px / p_distance;
  Jacobian(2, 0) = py * (vx_py - vy_px) / p_distance_cubed;
  Jacobian(2, 1) = px * (vy_px - vx_py) / p_distance_cubed;
  Jacobian(2, 2) = px / p_distance;
  Jacobian(2, 3) = py / p_distance;
  
  return Jacobian;
}
