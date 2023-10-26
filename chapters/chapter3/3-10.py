from tfx.extensions.google_cloud_big_query.example_gen.component import BigQueryExampleGen

query = """
    SELECT * FROM `<project_id>.<database>.<table_name>`
"""
example_gen = BigQueryExampleGen(query=query)
