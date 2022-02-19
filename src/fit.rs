use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::elastic_net::{ElasticNet, ElasticNetParameters};

pub fn elastic_net(feature: DenseMatrix<f64>, target: Vec<f64>, feat_for_pred: DenseMatrix<f64>) -> Vec<f64> {
    ElasticNet::fit(
        &feature,
        &target,
        ElasticNetParameters::default()
            .with_alpha(10.0)
            .with_l1_ratio(0.5),
    )
    .and_then(|rr| rr.predict(&feat_for_pred))
    .unwrap()
}
