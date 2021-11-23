#include <math.h>
#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);
  ekf_.x_ = VectorXd(4);
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.R_ = MatrixXd(2, 2);
  ekf_.H_ = MatrixXd(2, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
  
  H_laser_ << 1,0,0,0,
              0,1,0,0;

  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;
  
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;
  
  ekf_.Q_ << 0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {

    // first measurement
    cout << "EKF: " << endl;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      cout << "Initializing using RADAR measurement" << endl;
      ekf_.x_ << measurement_pack.raw_measurements_[0] * cos(measurement_pack.raw_measurements_[1]), 
                 measurement_pack.raw_measurements_[0] * sin(measurement_pack.raw_measurements_[1]), 
                 measurement_pack.raw_measurements_[2] * cos(measurement_pack.raw_measurements_[1]), 
                 measurement_pack.raw_measurements_[2] * sin(measurement_pack.raw_measurements_[1]);

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      cout << "Initializing using LIDAR measurement" << endl;
      ekf_.x_ << measurement_pack.raw_measurements_[0], 
                 measurement_pack.raw_measurements_[1], 
                 0, 
                 0;

    }

    // done initializing, no need to predict or update
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;
  
  float noise_ax_squared = pow(9.0, 2);
  float noise_ay_squared = pow(9.0, 2);
  ekf_.Q_(0, 0) = pow(dt, 4) * noise_ax_squared / 4;
  ekf_.Q_(1, 1) = pow(dt, 4) * noise_ay_squared / 4;
  ekf_.Q_(2, 2) = pow(dt, 2) * noise_ax_squared;
  ekf_.Q_(3, 3) = pow(dt, 2) * noise_ay_squared;
  ekf_.Q_(0, 2) = pow(dt, 3) * noise_ax_squared / 2;
  ekf_.Q_(2, 0) = pow(dt, 3) * noise_ax_squared / 2;
  ekf_.Q_(1, 3) = pow(dt, 3) * noise_ay_squared / 2;
  ekf_.Q_(3, 1) = pow(dt, 3) * noise_ay_squared / 2;

  ekf_.Predict();

  /**
   * Update
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    cout << "Updating using RADAR measurement" << endl;
    ekf_.Init(ekf_.x_, ekf_.P_, ekf_.F_, Hj_, R_radar_, ekf_.Q_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  } else {
    cout << "Updating using LIDAR measurement" << endl;
    ekf_.Init(ekf_.x_, ekf_.P_, ekf_.F_, H_laser_, R_laser_, ekf_.Q_);
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
