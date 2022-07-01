# Updating table properties in Hive within Spark

```python
desc = spark.sql("describe formatted <table_name>")
spark.sql("alter table <table_name> set tblproperties('ttl.table' = '180d')")
```

