import polars as pl

df = pl.read_csv(f'../data/frenchmtpl.csv', separator=';', infer_schema_length=0)

print(df)

schema_dict = {'IDpol': pl.String, 'ClaimNb': pl.Int8, 'Exposure': pl.Float64, 'VehPower': pl.Int8, 'VehAge': pl.Int8,
               'DrivAge': pl.Int8, 'BonusMalus': pl.Int16, 'VehBrand': pl.String, 'VehGas': pl.String, 'Area': pl.String,
               'Density': pl.Int16, 'Region': pl.String, 'ClaimAmount': pl.Float64}

df = df.cast(schema_dict)

df.write_csv(f'../data/frenchmtpl_clean.csv', separator=';')