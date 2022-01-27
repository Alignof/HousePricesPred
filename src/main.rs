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

fn df_to_dm(df: &DataFrame) -> Result<DenseMatrix<f64>> {
    let feature = df.select(vec![   
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
        "MiscVal",
    ])?
    .get_columns()
    .iter()
    .map(|ser| {
        if ser.dtype() == &DataType::Utf8 {
            ser.cast(&DataType::UInt64).unwrap()
        } else {
            ser.clone()
        }
    })
    .collect::<DataFrame>()
    .fill_null(FillNullStrategy::Mean)?;

    Ok(DenseMatrix::from_vec(
        feature.height(),
        feature.width(),
        &feature.to_ndarray::<Float64Type>()?.into_raw_vec(),
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

    let feature = df_to_dm(&df_train)?;
    let target = df_train
        .select(vec!["SalePrice"])?
        .to_ndarray::<Float64Type>()?
        .into_raw_vec();

    let test_feat = df_to_dm(&df_test)?;
    let test_ids = df_test
        .select(vec!["Id"])?
        .to_ndarray::<Float64Type>()?
        .into_raw_vec();

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
