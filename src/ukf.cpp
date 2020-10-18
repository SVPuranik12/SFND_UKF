#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;//30

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;//30
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(2*n_aug_ + 1);
  
  // state covariance matrix
 // P_.fill(0.0);
  
   P_ <<    1,   0,   0,   0,   0,
            0,   1,   0,   0,   0,
            0,   0,   1,   0,   0,
            0,   0,   0,   0.0225,   0,
            0,   0,   0,   0,   0.0225;
  
//   P_ <<
//     0.0054342,  -0.002405,  0.0034157, -0.0034819, -0.00299378,
//     -0.002405,    0.01084,   0.001492,  0.0098018,  0.00791091,
//     0.0034157,   0.001492,  0.0058012, 0.00077863, 0.000792973,
//    -0.0034819,  0.0098018, 0.00077863,   0.011923,   0.0112491,
//    -0.0029937,  0.0079109, 0.00079297,   0.011249,   0.0126972;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_,2*n_aug_ + 1);
  Xsig_pred_.fill(0.0);
  
  
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if(!is_initialized_)
  {
    
     if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
     {

      double rho = meas_package.raw_measurements_[0]; // range
      double phi = meas_package.raw_measurements_[1]; // bearing
      double rho_dot = meas_package.raw_measurements_[2]; //radial velocity
      double x = rho * std::cos(phi);
      double y = rho * std::sin(phi);
      double vx = rho_dot * std::cos(phi);
      double vy = rho_dot * std::sin(phi);
      double v = rho_dot;
       
     //Mapping of measurement space to predicted state is done here
      x_ << x , y, v, 0, 0;
     }
     else 
     {
       x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
     }
    
     //  saving previous timestamp in seconds. In this case initialised timestamp
      time_us_ = meas_package.timestamp_;
    
     // done initializing no need to predict or update
     is_initialized_ = true;
    
     return;
   }

   // calculate time difference for prediction
   double delta_T = (meas_package.timestamp_ - time_us_)/1000000.0;
   time_us_ = meas_package.timestamp_;

   // Prediction step
   Prediction(delta_T);

   // Update step
   if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
   {
     UpdateRadar(meas_package);
   }

   if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
   {
    UpdateLidar(meas_package);
   }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

    MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);
  
    // print Xsig_aug
    // std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;
  
    GenerateSigmaPoints(Xsig_aug);//Augmentation done here;
  
    // print Xsig_aug after generating sigma points
    //std::cout << "Xsig_aug after generating sigma points= " << std::endl << Xsig_aug << std::endl;
  
    PredictSigmaPoint(Xsig_aug, delta_t);
  
    // print Xsig_aug after sigma points prediction
    //std::cout << "Xsig_pred_ after sigma points prediction = " << std::endl << Xsig_pred_ << std::endl;
  
   std::cout << "P_" <<P_ << std::endl;
  
    PredictMeanAndCovariance();
}

void UKF::GenerateSigmaPoints(MatrixXd& Xsig_aug) 
{
  std::cout << "START:  GenerateSigmaPoints" << std::endl;
    
    //create augmented vectors

    //mean vector
    VectorXd x_aug = VectorXd::Zero(n_aug_);

    //state covariance
    MatrixXd p_aug = MatrixXd::Zero(n_aug_, n_aug_);

    //mean state
    x_aug.head(n_x_) = x_;

    //covariance matrix
    p_aug.topLeftCorner(n_x_, n_x_) = P_;
    p_aug(n_x_, n_x_) = std_a_ * std_a_;
    p_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

    // Matrix decomposition
    MatrixXd L = p_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i< n_aug_; i++) {
        Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
    }
  
  std::cout << "END:  GenerateSigmaPoints" << std::endl;
}


void UKF::PredictSigmaPoint(MatrixXd& Xsig_aug, double timediff)
{
  std::cout << "START:  PredictSigmaPoint" << std::endl;
    //predict sigma points
    for (int i = 0; i< 2*n_aug_+1; i++) {
        //extract values for better readability
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*timediff) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*timediff) );
        }
        else {
            px_p = p_x + v*timediff*cos(yaw);
            py_p = p_y + v*timediff*sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*timediff;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5*nu_a*timediff*timediff * cos(yaw);
        py_p = py_p + 0.5*nu_a*timediff*timediff * sin(yaw);
        v_p = v_p + nu_a*timediff;

        yaw_p = yaw_p + 0.5*nu_yawdd*timediff*timediff;
        yawd_p = yawd_p + nu_yawdd*timediff;

        //Copying Predicted sigma value to output parameter
        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = yaw_p;
        Xsig_pred_(4,i) = yawd_p;
    }
  
  std::cout << "END:  PredictSigmaPoint" << std::endl;
}


void UKF::PredictMeanAndCovariance() 
{
  std::cout << "START:  PredictMeanAndCovariance" << std::endl;
  
   weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++)
    {  //2n_aug_+1 weights
      double weight = 0.5 / (lambda_ + n_aug_);
      weights_(i) = weight;
    } 
  
    //State mean prediction
    x_.fill(0.0);
    
  //iterate over sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++)
    {
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

    std::cout << "END:  State mean prediction" << std::endl;
  
    //covariance matrix prediction
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

      std::cout << M_PI << std::endl;
      
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        P_ += weights_(i) * x_diff * x_diff.transpose() ;
      std::cout << "i =" << i << std::endl;
    }
  
  std::cout << "END:  PredictMeanAndCovariance" << std::endl;
}


void UKF::UpdateLidar(MeasurementPackage meas_package) 
{
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  //Lidar Dimension
  int n_z = 2;
  
  //Matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
  
  //Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  
  //Measurement Covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    
    VectorXd state_vec = Xsig_pred_.col(i);
    double px = state_vec(0);
    double py = state_vec(1);
    
    Zsig.col(i) << px,
                   py;
    
    //calculating mean predicted measurement
    z_pred = z_pred + (weights_(i) * Zsig.col(i));
  }
  
  //calculate measurement covariance matrix S
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {
    
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + (weights_(i) * z_diff * z_diff.transpose());
  }
  
  // Measurement Noise matrix
  MatrixXd R = MatrixXd(2,2);
  R << std_laspx_*std_laspx_,    0,
        0,                       std_laspy_*std_laspy_;
  
  // Predicted Measurement covariance matrix with Noise
  S = S + R;
  
  //LIDAR measurement (Ground Truth)
  VectorXd z = VectorXd(n_z);
  
  double meas_px = meas_package.raw_measurements_(0);
  double meas_py = meas_package.raw_measurements_(1);
  
  z << meas_px,
       meas_py;
  
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  
  //Cross Co-relation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    //normalize angles
    if (x_diff(3) > M_PI) 
    {
      x_diff(3) -= 2. * M_PI;
    } else if (x_diff(3) < -M_PI) 
    {
      x_diff(3) += 2. * M_PI;
    }
    
    VectorXd z_diff = Zsig.col(i) - z_pred;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();

  }
  
  // Residual vector
  VectorXd z_diff = z - z_pred;
   
  //calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();
  
  //Measurement Update: state mean and covariance matrix
  x_ = x_ + K*z_diff;
  P_ = P_ - K*S*K.transpose();
  
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  
  //RADAR Dimension
  int n_z = 3;
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
  
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  
  double rho = 0;
  double phi = 0;
  double rho_d = 0;
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {

    VectorXd state_vec = Xsig_pred_.col(i);
    double px = state_vec(0);
    double py = state_vec(1);
    double v = state_vec(2);
    double yaw = state_vec(3);
    double yaw_d = state_vec(4);
    
    rho = sqrt(px*px+py*py);
    phi = atan2(py,px);
    
    //Cater division by 0 error
    if(rho <0.001)
    {
      rho_d = (px*cos(yaw)*v+py*sin(yaw)*v) / 0.001;
    }
    else
    {
      rho_d = (px*cos(yaw)*v+py*sin(yaw)*v) / rho;
    }
    
    Zsig.col(i) << rho,
                   phi,
                   rho_d;
    
    //calculate mean predicted measurement
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  
  //calculate measurement covariance matrix S
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //Angle normalization
    if (z_diff(1) > M_PI)
    {
      z_diff(1) -= 2. * M_PI;
    } else if (z_diff(1) < - M_PI) 
    {
      z_diff(1) += 2. * M_PI;
    }
    S = S + (weights_(i) * z_diff * z_diff.transpose());
  }
  
  
  // Measurement noise matrix
  MatrixXd R = MatrixXd(3,3);
  R << std_radr_*std_radr_,            0,             0,
       0,                    std_radphi_*std_radphi_, 0,
       0,                    0,                       std_radrd_*std_radrd_;
 
  // Predicted Measurement Covariance matrix with Noise
  S = S + R;
  
  //RADAR Measurement
  VectorXd z = VectorXd(n_z);
  
  double rho_meas = meas_package.raw_measurements_(0);
  double phi_meas = meas_package.raw_measurements_(1);
  double rhod_meas = meas_package.raw_measurements_(2);
  
  z << rho_meas,
       phi_meas,
       rhod_meas;
  
  //cross correlation Tc matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  
  //calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    //normalize angles
    if (x_diff(3) > M_PI)
    {
      x_diff(3) -= 2. * M_PI;
    } else if (x_diff(3) < -M_PI) 
    {
      x_diff(3) += 2. * M_PI;
    }
    
    //Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    //normalize angles
    if (z_diff(1) > M_PI) 
    {
      z_diff(1) -= 2. * M_PI;
    } else if (z_diff(1) < -M_PI)
    {
      z_diff(1) += 2. * M_PI;
    }
    
    Tc = Tc + (weights_(i) * x_diff * z_diff.transpose());
    
  }
  
  // residual
  VectorXd z_diff = z - z_pred;
  
  //normalize angles
  if (z_diff(1) > M_PI)
  {
    z_diff(1) -= 2. * M_PI;
  } else if (z_diff(1) < -M_PI)
  {
    z_diff(1) += 2. * M_PI;
  }
  
  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  
  //update state mean and covariance matrix
  x_ = x_ + K*z_diff;
  P_ = P_ - K*S*K.transpose();
}