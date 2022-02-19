fn elastic_net(feature: DenseMatrix, target: DenseMatrix) {
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
