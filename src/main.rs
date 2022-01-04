use polars::prelude::*;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;

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
    let target_df = df.select(vec!["Id"])?;

    let feature = DenseMatrix::from_vec(
        feature_df.height(),
        feature_df.width(),
        &feature_df.to_ndarray::<Float64Type>()?.into_raw_vec()
    );

    dbg!(feature);

    Ok(())
}

