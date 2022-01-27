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
    df.select(vec![   
        "Neighborhood",
        /*
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
        "GarageCond",
        "Fence",
        "SaleType",
        "SaleCondition",
        "LotFrontage",
        "LotArea",
        "OverallQual",
        "OverallCond",
        "MasVnrArea",
        */
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
        "MiscVal",
    ])?
    .get_columns()
    .iter()
    .flat_map(|ser| {
        match ser.dtype() {
            DataType::Utf8 => {
                match ser.strict_cast(&DataType::UInt64) {
                    Ok(ser) => vec!(ser),
                    Err(_) => ser.to_dummies().unwrap().get_columns().to_vec(),
                }
            },
            _ => vec!(ser.clone())
        }
    })
    .collect::<DataFrame>()
    .fill_null(FillNullStrategy::Mean)
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

    let feat_train = select_feature(&df_train)?;
    let feat_test = select_feature(&df_test)?;

    let feature = df_to_dm(&feat_train)?;
    let test_feat = df_to_dm(&feat_test)?;
    let rr_predicted = RidgeRegression::fit(
        &feature,
        &target,
        RidgeRegressionParameters::default().with_alpha(5.0),
    )
    .and_then(|rr| rr.predict(&test_feat))
    .unwrap();

    save_predict(test_ids, rr_predicted);

    Ok(())
}
