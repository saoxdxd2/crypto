import polars as pl
import pandera.polars as pa

class TestModel(pa.DataFrameModel):
    a: int = pa.Field()
    b: int = pa.Field()

    @pa.dataframe_check
    @classmethod
    def check_ab(cls, df: pl.DataFrame) -> pl.Series:
        return df["a"] < df["b"]

df = pl.DataFrame({"a": [1, 2], "b": [2, 3]})
print("Validating with @classmethod...")
TestModel.validate(df)
print("Success!")
