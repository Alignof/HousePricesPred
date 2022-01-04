use polars::prelude::*;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::ridge_regression::{RidgeRegression, RidgeRegressionParameters};

fn load_df() -> Result<(DataFrame, DataFrame)> {
    let test_path = "data/test.csv";
    let df_test = CsvReader::from_path(test_path)?
        .infer_schema(None)
        .with_delimiter(b',')
        .has_header(true)
        .finish()?;

    let train_path = "data/train.csv";
    let df_train = CsvReader::from_path(train_path)?
        .infer_schema(None)
        .with_delimiter(b',')
        .has_header(true)
        .finish()?;

    Ok((df_train, df_test))
}

fn main() -> Result<()> {
    let (df_train, df_test) = load_df().expect("load csv error.");
    let selected_feature = vec![
        "LotArea",
        "OverallQual",
        "OverallCond",
        "TotalBsmtSF",
        "1stFlrSF",
        "GarageCars",
    ];

    let feature = df_train.select(&selected_feature)?;
    let feature = DenseMatrix::from_vec(
        feature.height(),
        feature.width(),
        &feature.to_ndarray::<Float64Type>()?.into_raw_vec()
    );
    let target = df_train.select(vec!["SalePrice"])?
        .to_ndarray::<Float64Type>()?
        .into_raw_vec();

    let feature_test = df_test.select(&selected_feature)?;
    let feature_test = DenseMatrix::from_vec(
        feature_test.height(),
        feature_test.width(),
        &feature_test.to_ndarray::<Float64Type>()?.into_raw_vec()
    );

    let rr_predicted = RidgeRegression::fit(
        &feature,
        &target,
        RidgeRegressionParameters::default().with_alpha(0.8),
    )
    .and_then(|rr| rr.predict(&feature_test))
    .unwrap();

    dbg!(&rr_predicted);

    Ok(())
}

