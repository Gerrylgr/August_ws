import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterType
from rcl_interfaces.srv import SetParameters, GetParameters
import threading
import time
import sounddevice as sd
from MeloTTS.melo.api import TTS     # 注意文件结构，只有有__init__.py才会被认为是python包;并且必须这样导入，否则会和源码导入方式不一致导致后边报错
import queue
from functools import partial       # 用于在回调函数里传入多个变量
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from ament_index_python.packages import get_package_share_directory

from cleaned_message_interface.msg import CleanedMessage

def init_model(config_path, ckpt_path, language='ZH', device='cuda:0', use_hf=False):
    model = TTS(
        language=language,
        device=device,
        use_hf=use_hf,  # 关键！要禁用 huggingface 自动下载
        config_path=config_path,
        ckpt_path=ckpt_path,
    )
    return model

class MeloSpeaker(Node):
    def __init__(self, model, speed):
        super().__init__('melo_speaker')
        self.target_node_name = '/signals_saver'
        self.client_set = self.create_client(SetParameters, self.target_node_name + '/set_parameters')
        self.client_get = self.create_client(GetParameters, self.target_node_name + '/get_parameters')

        self.MyQueue = queue.Queue()

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50
        )
        self.answer_subscriber_ = self.create_subscription(
                                                CleanedMessage, 
                                                'ai_answer', 
                                                partial(self.melo_callback, model, speed), 
                                                qos_profile)

    def melo_callback(self, model, speed, msg):
        if not msg.cleaned_message.strip():
            print("MESSAGE IS NOT VALID!!!")
            print(msg.cleaned_message)
            # return
        else:
            self.get_logger().info("Valid MSG Received!!!")
            audio_data = model.tts_to_file(
                msg.cleaned_message,
                speaker_id=model.hps.data.spk2id['ZH'],
                output_path=None,  # 传 None，就会返回 numpy array
                speed=speed,
            )
            self.MyQueue.put(audio_data)


    def try_set_param(self, param_name, param_value):
        while not self.client_set.wait_for_service(timeout_sec=0.2):
            self.get_logger().info('Waiting for parameter service...')
            time.sleep(0.1)
            # return False

        self.get_logger().info(f'Sending parameter update: {param_name} = {param_value}')

        req = SetParameters.Request()
        param = Parameter()
        param.name = param_name
        param.value.type = ParameterType.PARAMETER_BOOL
        param.value.bool_value = param_value
        req.parameters = [param]

        future = self.client_set.call_async(req)

        rclpy.spin_until_future_complete(self, future, timeout_sec=20.0)
        
        if future.done() and future.result() is not None:
            self.get_logger().info(f"Setting PARAM {param_name} successful!!!")
            return future.result().results[0].successful
        else:
            self.get_logger().warn(f"Setting PARAM {param_name} FAILED !!!")
            return False

    def try_get_param(self, param_name):
        last_time = time.time()
        while not self.client_get.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for parameter service...')
            time.sleep(1)

        self.get_logger().info('Parameter service is available. Requesting parameter value...')

        request = GetParameters.Request()
        request.names = [param_name]
        future = self.client_get.call_async(request)

        rclpy.spin_until_future_complete(self, future, timeout_sec=20.0)

        if future.done() and future.result() is not None:
            self.get_logger().info(f"Getting PARAM {param_name} successful!!!")
            param_value = future.result().values[0]

            if param_value.type == ParameterType.PARAMETER_BOOL:
                return param_value.bool_value
            elif param_value.type == ParameterType.PARAMETER_INTEGER:
                return param_value.integer_value
            elif param_value.type == ParameterType.PARAMETER_DOUBLE:
                return param_value.double_value
            elif param_value.type == ParameterType.PARAMETER_STRING:
                return param_value.string_value
            else:
                self.get_logger().warn(f"Unsupported parameter type: {param_value.type}")
                return None
        else:
            self.get_logger().warn(f"Getting PARAM {param_name} FAILED !!!")
            return None

    def consumer(self, model):
        SIGNAL = True       # SIGNAL可以保证self.try_set_param('consumer_signal', True)只在每次回答最后运行一次
        while True:
            # rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.05)
            if not self.MyQueue.empty():
                print("QUEUE NOT EMPTY!!!")
                SIGNAL = True
                audio_data = self.MyQueue.get()
                if not self.try_get_param('tts_signal'):
                    # 创建一个子线程，先播放音频，再等待这段音频播放完成
                    # threading.Thread(target=lambda: (sd.play(audio_data, samplerate=model.hps.data.sampling_rate), sd.wait()), daemon=True).start()
                    sd.play(audio_data, samplerate=model.hps.data.sampling_rate)
                    sd.wait()
                else:
                    time.sleep(0.01)
                    print("MeloTest::检测到中断，停止播放！！！")
            else:
                if SIGNAL:
                    self.try_set_param('consumer_signal', True)
                    SIGNAL = False

def main(args=None):
    package_path = get_package_share_directory('august_cerebra')

    speed = 0.85
    device = 'cuda:0' 

    config_path = package_path + '/MeloTTS/melo/configs/config.json'
    ckpt_path = package_path + '/MeloTTS/melo/configs/checkpoint.pth'

    model = init_model(config_path, ckpt_path, device=device)
    print("Model Loaded!")

    rclpy.init(args=args)
    MeloSpeaker_node = MeloSpeaker(model, speed)
    # MeloSpeaker_node.try_set_param('tts_signal', False)

    consumer_thread = threading.Thread(target=MeloSpeaker_node.consumer, args=(model,))
    consumer_thread.start()

    rclpy.spin(MeloSpeaker_node)
    rclpy.shutdown()