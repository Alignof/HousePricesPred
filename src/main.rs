use polars::prelude::*;

fn main() -> Result<()> {
    let train_path = "data/train.csv";
    let df = CsvReader::from_path(train_path)?
        .infer_schema(None)
        .with_delimiter(b',')
        .has_header(true)
        .finish()?;

    let train_df = df.select(
        vec![
            "Id",
            "LotArea",
            "OverallQual",
            "OverallCond",
            "TotalBsmtSF",
            "1stFlrSF",
            "GarageCars",
        ]        
    )?;

    dbg!(train_df);

    Ok(())
}

