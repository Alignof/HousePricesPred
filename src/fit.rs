use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::elastic_net::{ElasticNet, ElasticNetParameters};
use smartcore::linear::ridge_regression::{RidgeRegression, RidgeRegressionParameters};
use smartcore::svm::{
    Kernels,
    svr::{SVRParameters, SVR},
};
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;

#[allow(dead_code)]
pub fn elastic_net(feature: DenseMatrix<f64>, target: Vec<f64>, feat_for_pred: DenseMatrix<f64>) -> Vec<f64> {
    ElasticNet::fit(
        &feature,
        &target,
        ElasticNetParameters::default()
            .with_alpha(60.0)
            .with_l1_ratio(0.7)
    )
    .and_then(|lr| lr.predict(&feat_for_pred))
    .unwrap()
}


#[allow(dead_code)]
pub fn ridge_regression(feature: DenseMatrix<f64>, target: Vec<f64>, feat_for_pred: DenseMatrix<f64>) -> Vec<f64> {
    RidgeRegression::fit(
        &feature,
        &target,
        RidgeRegressionParameters::default().with_alpha(10.0)
    )
    .and_then(|rr| rr.predict(&feat_for_pred))
    .unwrap()
}

#[allow(dead_code)]
pub fn support_vector_regressor(feature: DenseMatrix<f64>, target: Vec<f64>, feat_for_pred: DenseMatrix<f64>) -> Vec<f64> {
    SVR::fit(
        &feature,
        &target,
        SVRParameters::default()
            .with_kernel(Kernels::rbf(0.5))
            .with_c(2000.0)
            .with_eps(10.0)
    )
    .and_then(|svm| svm.predict(&feat_for_pred))
    .unwrap()
}

#[allow(dead_code)]
pub fn random_forest(feature: DenseMatrix<f64>, target: Vec<f64>, feat_for_pred: DenseMatrix<f64>) -> Vec<f64> {
    RandomForestRegressor::fit(
        &feature,
        &target,
        Default::default()
    )
    .and_then(|rf| rf.predict(&feat_for_pred))
    .unwrap()
}


