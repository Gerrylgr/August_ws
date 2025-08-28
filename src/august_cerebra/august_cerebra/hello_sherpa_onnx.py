import sys
from pathlib import Path
import sherpa_onnx
import sounddevice as sd

from ament_index_python.packages import get_package_share_directory

def assert_file_exists(path):
    if not Path(path).is_file():
        print(f"❌ 文件不存在: {path}")
        sys.exit(-1)

def initialize_recognizer(model_dir):
    tokens = f"{model_dir}/tokens.txt"
    encoder = f"{model_dir}/encoder-epoch-99-avg-1.onnx"
    decoder = f"{model_dir}/decoder-epoch-99-avg-1.onnx"
    joiner = f"{model_dir}/joiner-epoch-99-avg-1.onnx"

    for f in [tokens, encoder, decoder, joiner]:
        assert_file_exists(f)

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
        decoding_method="greedy_search",
        provider="cuda",  # 如果你有 GPU 可以改成 "cuda"
    )
    return recognizer

def speech_recognition(recognizer, sample_rate=48000, samples_per_read=4800, if_interrupt=False):
    stream = recognizer.create_stream()
    last_result = ""
    segment_id = 0

    try:
        with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
            while True:
                samples, _ = s.read(samples_per_read)
                samples = samples.reshape(-1)
                stream.accept_waveform(sample_rate, samples)

                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                result = recognizer.get_result(stream)
                is_endpoint = recognizer.is_endpoint(stream)

                if result and result != last_result:
                    last_result = result
                    if not if_interrupt:
                        print(f"\r{segment_id}: {result}", end="", flush=True)

                if is_endpoint:
                    if result:
                        if not if_interrupt:
                            print(f"\r{segment_id}: {result}", flush=True)
                        segment_id += 1
                        return result  # 返回识别的文本，如果不return就可以一直读取
                    recognizer.reset(stream)

    except KeyboardInterrupt:
        print("\n🛑 语音识别结束。感谢使用！")
        return None

if __name__ == "__main__":
    # 使用示例
    # model_dir = "F:\\PythonProject\\RoboNova_Lib\\sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
    model_dir = get_package_share_directory('august_cerebra') + '/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20'
    recognizer = initialize_recognizer(model_dir)
    while True:
        recognized_text = speech_recognition(recognizer)
        print(f"识别结果: {recognized_text}")

