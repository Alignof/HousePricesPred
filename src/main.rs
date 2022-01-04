use polars::prelude::*;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::ridge_regression::{RidgeRegression, RidgeRegressionParameters};
use smartcore::model_selection::train_test_split;
use smartcore::metrics::mean_squared_error;

fn main() -> Result<()> {
    let train_path = "data/train.csv";
    let df = CsvReader::from_path(train_path)?
        .infer_schema(None)
        .with_delimiter(b',')
        .has_header(true)
        .finish()?;

    let feature_df = df.select(
        vec![
            "LotArea",
            "OverallQual",
            "OverallCond",
            "TotalBsmtSF",
            "1stFlrSF",
            "GarageCars",
        ]        
    )?;
    let feature = DenseMatrix::from_vec(
        feature_df.height(),
        feature_df.width(),
        &feature_df.to_ndarray::<Float64Type>()?.into_raw_vec()
    );
    let target = df.select(vec!["Id"])?
        .to_ndarray::<Float64Type>()?
        .into_raw_vec();

    let (f_train, f_test, t_train, t_test) = train_test_split(&feature, &target, 0.2, true);

    let rr_predicted = RidgeRegression::fit(
        &f_train,
        &t_train,
        RidgeRegressionParameters::default().with_alpha(0.8),
    )
    .and_then(|rr| rr.predict(&f_test))
    .unwrap();

    dbg!(&rr_predicted);

    println!(
        "RMSE Ridge Regression: {}",
        mean_squared_error(&t_test, &rr_predicted).cbrt()
    );

    Ok(())
}

