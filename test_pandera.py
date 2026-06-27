import polars as pl
import pandera.polars as pa

class TestModel(pa.DataFrameModel):
    a: int = pa.Field()
    b: int = pa.Field()

    @pa.dataframe_check
    @classmethod
    def check_ab(cls, data: Any) -> pl.LazyFrame:
        return data.lazyframe.select(pl.col("a") < pl.col("b"))

df = pl.DataFrame({"a": [1, 2], "b": [2, 3]})
print("Validating with @classmethod...")
TestModel.validate(df)
print("Success!")
