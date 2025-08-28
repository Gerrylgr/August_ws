import threading
import time
import sounddevice as sd
from MeloTTS.melo.api import TTS
import queue
from ament_index_python.packages import get_package_share_directory

tts_signal = False
consumer_signal = False

MyQueue = queue.Queue()

def init_model(config_path, ckpt_path, language='ZH', device='cuda:0', use_hf=False):
    model = TTS(
        language=language,
        device=device,
        use_hf=use_hf,  # 关键！要禁用 huggingface 自动下载
        config_path=config_path,
        ckpt_path=ckpt_path,
    )
    return model

def processor(text, model, output_path=None, speed=0.85):
    audio_data = model.tts_to_file(
        text,
        speaker_id=model.hps.data.spk2id['ZH'],
        output_path=output_path,  # 传 None，就会返回 numpy array
        speed=speed,
    )
    # sd.play(audio_data, samplerate=model.hps.data.sampling_rate)
    # sd.wait()
    MyQueue.put(audio_data)

def consumer(model):
    while True:
        time.sleep(0.1)
        if not MyQueue.empty():
            audio_data = MyQueue.get()
            if not tts_signal:
                sd.play(audio_data, samplerate=model.hps.data.sampling_rate)
                sd.wait()
            else:
                time.sleep(0.01)
                print("MeloTest::检测到中断，停止播放！！！")
        else:
            global consumer_signal
            consumer_signal = True

if __name__ == '__main__':
    package_path = get_package_share_directory('august_cerebra')

    # 合成参数
    speed = 0.85
    device = 'cuda:0'  # 或者 'cpu'

    config_path = package_path + '/MeloTTS/melo/configs/config.json'
    ckpt_path = package_path + '/MeloTTS/melo/configs/checkpoint.pth'

    model = init_model(config_path, ckpt_path, device=device)
    print("Model Loaded!")

    consumer_thread = threading.Thread(target=consumer, args=(model,)).start()

    while True:
        text1 = "你好，这是一个测试音频。"
        processor(text1, model, speed=speed)
        text2 = "开发者需要共享的坐标系约定，"
        processor(text2, model, speed=speed)
        text3 = "以更好地集成和重用驱动程序、模型和库等软件组件。"
        processor(text3, model, speed=speed)
        text4 = "这个共享的坐标系约定可为创建移动基座的驱动程序和模型的开发者提供规范。"
        processor(text4, model, speed=speed)
        text5 = "To be, or not to be: that is the question."
        processor(text5, model, speed=speed)
        text6 = "《将进酒》是唐代著名浪漫主义诗人李白沿用乐府古题创作的一首七言歌行。"
        processor(text6, model, speed=speed)
        text7 = "君不见黄河之水天上来，奔流到海不复回。"
        processor(text7, model, speed=speed)
        text8 = "人生得意须尽欢，莫使金樽空对月。"
        processor(text8, model, speed=speed)
        text9 = "烹羊宰牛且为乐，会须一饮三百杯。"
        processor(text9, model, speed=speed)
        time.sleep(3)