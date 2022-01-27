use std::fs::File;
use std::io::Write;
use polars::prelude::*;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::ridge_regression::{RidgeRegression, RidgeRegressionParameters};

fn csv_to_df(file_path: &str) -> Result<DataFrame> {
    CsvReader::from_path(file_path)?
        .with_n_threads(Some(4))
        .with_null_values(Some(NullValues::AllColumns("NA".to_string())))
        .with_delimiter(b',')
        .infer_schema(None)
        .has_header(true)
        .finish()
}

fn select_feature(df: &DataFrame) -> Result<DataFrame> {
    Ok(df.select(vec![   
        "Neighborhood",
        "MSZoning",
        "Utilities",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "ExterQual",
        "ExterCond",
        "Foundation",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "Heating",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
        "GarageType",
        "GarageQual",
        "SaleType",
        "GarageCond",
        "Fence",
        "SaleCondition",
        "LotFrontage",
        "LotArea",
        "OverallQual",
        "OverallCond",
        "MasVnrArea",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "FullBath",
        "BedroomAbvGr",
        "KitchenAbvGr",
        "TotRmsAbvGrd",
        "GarageArea",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
    ])?
    .get_columns()
    .iter()
    .map(|ser| {
        match ser.dtype() {
            DataType::Utf8 => ser.strict_cast(&DataType::Int64).unwrap_or(ser.clone()),
            _ => ser.clone(),
        }
    })
    .collect::<DataFrame>())
}

fn get_features_df(train: DataFrame, test: DataFrame) -> Result<(DataFrame, DataFrame)> {
    let feat_train = select_feature(&train)?;
    let feat_test = select_feature(&test)?;
    let features = feat_train.vstack(&feat_test)?
        .get_columns()
        .iter()
        .flat_map(|ser| {
            match ser.dtype() {
                DataType::Utf8 => {
                    ser.to_dummies()
                        .unwrap()
                        .get_columns()
                        .iter()
                        .filter(|ser| ser.sum() != Some(0))
                        .map(|ser| ser.clone())
                        .collect::<Vec<Series>>()
                }
                _ => vec!(ser.clone())
            }
        })
        .collect::<DataFrame>()
        .fill_null(FillNullStrategy::Mean)?;

    Ok((features.head(Some(train.height())), features.tail(Some(test.height()))))
}

fn df_to_dm(df: &DataFrame) -> Result<DenseMatrix<f64>> {
    Ok(DenseMatrix::from_vec(
        df.height(),
        df.width(),
        &df.to_ndarray::<Float64Type>()?.into_raw_vec(),
    ))
}

fn save_predict(ids: Vec<f64>, pred: Vec<f64>) {
    let save_path = "data/submission.csv";
    let mut file = File::create(save_path).unwrap();

    writeln!(&mut file, "Id,SalePrice").unwrap();
    for (id, price) in ids.iter().zip(pred.iter()) {
        writeln!(&mut file, "{},{}", id, price).unwrap();
    }
}

fn main() -> Result<()> {
    let df_test = csv_to_df("data/test.csv")?;
    let df_train = csv_to_df("data/train.csv")?;

    let target = df_train
        .select(vec!["SalePrice"])?
        .to_ndarray::<Float64Type>()?
        .into_raw_vec();

    let test_ids = df_test
        .select(vec!["Id"])?
        .to_ndarray::<Float64Type>()?
        .into_raw_vec();

    let (feat_train, feat_test) = get_features_df(df_train, df_test)?;
    dbg!(&feat_train[27].sum::<i32>());

    let feature = df_to_dm(&feat_train)?;
    let feat_for_pred = df_to_dm(&feat_test)?;
    let rr_predicted = RidgeRegression::fit(
        &feature,
        &target,
        RidgeRegressionParameters::default().with_alpha(0.1),
    )
    .and_then(|rr| rr.predict(&feat_for_pred))
    .unwrap();

    save_predict(test_ids, rr_predicted);

    Ok(())
}
