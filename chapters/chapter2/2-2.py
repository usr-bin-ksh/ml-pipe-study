import re

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

input_file = "gs://dataflow-samples/shakespeare/kinglear.txt"
output_file = "/tmp/output.txt"

# 파이프라인의 옵션 객체를 정의
pipeline_options = PipelineOptions()

with beam.Pipeline(options=pipeline_options) as p:
    # 텍스트 파일을 읽거나 파일 패턴을 PCollection으로 변환
    lines = p | ReadFromText(input_file)

    # 각 단어의 등장 횟수
    counts = (
            lines
            | 'Split' >> beam.FlatMap(lambda x: re.findall(r'[A-Za-z\']+', x))
            | 'PairWithOne' >> beam.Map(lambda x: (x, 1))
            | 'GroupAndSum' >> beam.CombinePerKey(sum))

    # 각 단어의 등장 횟수를 문자열로 변환해 PCollection 에 저장
    def format_result(word_count):
        (word, count) = word_count
        return "{}: {}".format(word, count)

    output = counts | 'Format' >> beam.Map(format_result)

    # “Write” 트랜스폼 명령으로 결과를 출력
    output | WriteToText(output_file)
