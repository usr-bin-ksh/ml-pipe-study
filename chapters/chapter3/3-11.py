from tfx.examples.custom_components.presto_example_gen.proto import presto_config_pb2
from tfx.examples.custom_components.presto_example_gen.presto_component.component import PrestoExampleGen

query = """
    SELECT * FROM `<project_id>.<database>.<table_name>`
"""
presto_config = presto_config_pb2.PrestoConnConfig(
    host='localhost',
    port=8080)
example_gen = PrestoExampleGen(presto_config, query=query)
