use polars::prelude::*;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::ridge_regression::{RidgeRegression, RidgeRegressionParameters};

fn csv_to_df(file_path: &str) -> Result<DataFrame> {
    Ok(
        CsvReader::from_path(file_path)?
            .infer_schema(None)
            .with_delimiter(b',')
            .has_header(true)
            .finish()?
    )
}

fn df_to_dm(df: &DataFrame) -> Result<DenseMatrix<f64>> {
    let selected_feature = vec![
        "LotArea",
        "OverallQual",
        "OverallCond",
        "TotalBsmtSF",
        "1stFlrSF",
        "GarageCars",
    ];
    let feature = df.select(&selected_feature)?;

    Ok(
        DenseMatrix::from_vec(
            feature.height(),
            feature.width(),
            &feature.to_ndarray::<Float64Type>()?.into_raw_vec()
        )
    )
}

fn main() -> Result<()> {
    let df_test = csv_to_df("data/test.csv")?;
    let df_train = csv_to_df("data/train.csv")?;

    let feature = df_to_dm(&df_train)?;
    let feature_test = df_to_dm(&df_test)?;
    let target = df_train.select(vec!["SalePrice"])?
        .to_ndarray::<Float64Type>()?
        .into_raw_vec();


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

