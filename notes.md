notes:
 * for redshift, do: UNLOAD (subquery) to temp s3 then stream that vs whatever dataio is doing via api???
 * for cassandra pd.read_sql(query, con=cassandra_con, chunksize=N) assuming it's compliant
 * trivially parallelize via asyncio, httpx
 * circ stats --> scipy once my pr is in for irwinhall -- rao's test OR use cramer-vonMises in scipy with rayleigh (not worth it)
 * general data retrieval hooks -> airflow, imo
 * geodata -> use Geopandas for viz/calc
 * separate viz from data compute

nits:
 * ci for docs and such
 * ruff lint, mypy types, whole 9 yards
 * poetry, tests, gh actions etc
 * docker deployment
 * structured logging
