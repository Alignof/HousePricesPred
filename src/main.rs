use polars::prelude::*;

use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::ridge_regression::{RidgeRegression, RidgeRegressionParameters};
// Model performance
use smartcore::metrics::mean_squared_error;
use smartcore::model_selection::train_test_split;

fn main() {
    let data_path = "data/train.csv";
}

fn get_schema() -> Schema {
    Schema::new(vec![
        Field::new("Id",			DataType::UInt32),
        Field::new("MSSubClass",	DataType::UInt32),
        Field::new("MSZoning",		DataType::UInt32),
        Field::new("LotFrontage",	DataType::UInt32),
        Field::new("LotArea",		DataType::UInt32),
        Field::new("Street",		DataType::UInt32),
        Field::new("Alley",			DataType::UInt32),
        Field::new("LotShape",		DataType::UInt32),
        Field::new("LandContour",	DataType::UInt32),
        Field::new("Utilities",		DataType::UInt32),
        Field::new("LotConfig",		DataType::UInt32),
        Field::new("LandSlope",		DataType::UInt32),
        Field::new("Neighborhood",	DataType::UInt32),
        Field::new("Condition1",	DataType::UInt32),
        Field::new("Condition2",	DataType::UInt32),
        Field::new("BldgType",		DataType::UInt32),
        Field::new("HouseStyle",	DataType::UInt32),
        Field::new("OverallQual",	DataType::UInt32),
        Field::new("OverallCond",	DataType::UInt32),
        Field::new("YearBuilt",		DataType::UInt32),
        Field::new("YearRemodAdd",	DataType::UInt32),
        Field::new("RoofStyle",		DataType::UInt32),
        Field::new("RoofMatl",		DataType::UInt32),
        Field::new("Exterior1st",	DataType::UInt32),
        Field::new("Exterior2nd",	DataType::UInt32),
        Field::new("MasVnrType",	DataType::UInt32),
        Field::new("MasVnrArea",	DataType::UInt32),
        Field::new("ExterQual",		DataType::UInt32),
        Field::new("ExterCond",		DataType::UInt32),
        Field::new("Foundation",	DataType::UInt32),
        Field::new("BsmtQual",		DataType::UInt32),
        Field::new("BsmtCond",		DataType::UInt32),
        Field::new("BsmtExposure",	DataType::UInt32),
        Field::new("BsmtFinType1",	DataType::UInt32),
        Field::new("BsmtFinSF1",	DataType::UInt32),
        Field::new("BsmtFinType2",	DataType::UInt32),
        Field::new("BsmtFinSF2",	DataType::UInt32),
        Field::new("BsmtUnfSF",		DataType::UInt32),
        Field::new("TotalBsmtSF",	DataType::UInt32),
        Field::new("Heating",		DataType::UInt32),
        Field::new("HeatingQC",		DataType::UInt32),
        Field::new("CentralAir",	DataType::UInt32),
        Field::new("Electrical",	DataType::UInt32),
        Field::new("1stFlrSF",		DataType::UInt32),
        Field::new("2ndFlrSF",		DataType::UInt32),
        Field::new("LowQualFinSF",	DataType::UInt32),
        Field::new("GrLivArea",		DataType::UInt32),
        Field::new("BsmtFullBath",	DataType::UInt32),
        Field::new("BsmtHalfBath",	DataType::UInt32),
        Field::new("FullBath",		DataType::UInt32),
        Field::new("HalfBath",		DataType::UInt32),
        Field::new("BedroomAbvGr",	DataType::UInt32),
        Field::new("KitchenAbvGr",	DataType::UInt32),
        Field::new("KitchenQual",	DataType::UInt32),
        Field::new("TotRmsAbvGrd",	DataType::UInt32),
        Field::new("Functional",	DataType::UInt32),
        Field::new("Fireplaces",	DataType::UInt32),
        Field::new("FireplaceQu",	DataType::UInt32),
        Field::new("GarageType",	DataType::UInt32),
        Field::new("GarageYrBlt",	DataType::UInt32),
        Field::new("GarageFinish",	DataType::UInt32),
        Field::new("GarageCars",	DataType::UInt32),
        Field::new("GarageArea",	DataType::UInt32),
        Field::new("GarageQual",	DataType::UInt32),
        Field::new("GarageCond",	DataType::UInt32),
        Field::new("PavedDrive",	DataType::UInt32),
        Field::new("WoodDeckSF",	DataType::UInt32),
        Field::new("OpenPorchSF",	DataType::UInt32),
        Field::new("EnclosedPorch"  DataType::UInt32),
        Field::new("3SsnPorch",		DataType::UInt32),
        Field::new("ScreenPorch",	DataType::UInt32),
        Field::new("PoolArea",		DataType::UInt32),
        Field::new("PoolQC",		DataType::UInt32),
        Field::new("Fence",			DataType::UInt32),
        Field::new("MiscFeature",	DataType::UInt32),
        Field::new("MiscVal",		DataType::UInt32),
        Field::new("MoSold",		DataType::UInt32),
        Field::new("YrSold",		DataType::UInt32),
        Field::new("SaleType",		DataType::UInt32),
        Field::new("SaleCondition"  DataType::UInt32),
        Field::new("SalePrice",		DataType::UInt32),
    ])
}

DataType::UInt32
DataType::UInt32
RL
DataType::UInt32
DataType::UInt32
Pave
NA
Reg
Lvl
AllPub
Inside
Gtl
CollgCr
Norm
Norm
1Fam
2Story
DataType::UInt32
DataType::UInt32
DataType::UInt32
DataType::UInt32
Gable
CompShg
VinylSd
VinylSd
BrkFace
DataType::UInt32
Gd
TA
PConc
Gd
TA
No
GLQ
DataType::UInt32
Unf
DataType::UInt32
DataType::UInt32
DataType::UInt32
GasA
Ex
Y
SBrkr
DataType::UInt32
DataType::UInt32
DataType::UInt32
DataType::UInt32
DataType::UInt32
DataType::UInt32
DataType::UInt32
DataType::UInt32
DataType::UInt32
DataType::UInt32
Gd
DataType::UInt32
Typ
DataType::UInt32
NA
Attchd
DataType::UInt32
RFn
DataType::UInt32
DataType::UInt32
TA
TA
Y
DataType::UInt32
DataType::UInt32
DataType::UInt32
DataType::UInt32
DataType::UInt32
DataType::UInt32
NA
NA
NA
DataType::UInt32
DataType::UInt32
DataType::UInt32
WD
Normal
DataType::UInt32
