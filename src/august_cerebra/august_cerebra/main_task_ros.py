import rclpy
from rclpy.node import Node
from cleaned_message_interface.msg import CleanedMessage
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
from sensor_msgs.msg import Image
from rcl_interfaces.msg import Parameter, ParameterType
from rcl_interfaces.srv import SetParameters, GetParameters
from rclpy.qos import QoSProfile
from tf_transformations import euler_from_quaternion
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
import base64
import re
import threading
import time
import difflib
import jieba
from json import dumps, loads
import json
from threading import Thread
import cv2
import requests
from playsound import playsound
from august_cerebra import hello_sherpa_onnx

tts_stop_event = threading.Event()        # 用于控制是否中断朗读

package_path = get_package_share_directory('august_cerebra')
model_dir = package_path + '/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20'
FILE_PATH = package_path + "/knowledge_history/knowledge.txt"  #知识库路径
token = "ollama"

# 把 Quaternion 转换成 list
def quaternion_to_euler(quat):
    return euler_from_quaternion([
        quat.x,
        quat.y,
        quat.z,
        quat.w
    ])


"""文本读写函数"""
def read_file_content(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading file: {e}")
        return ""

def write_file_content(file_path: str, content: str):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
    except IOError as e:
        print(f"Error writing file: {e}")

"""判断是否为唤醒词"""
def is_wake_word_in_sentence(sentence, wake_words, threshold=0.5):
    sentence = sentence.lower()

    # 中文分词
    words = jieba.lcut(sentence)

    for word in words:
        print(word)
        matches = difflib.get_close_matches(word, wake_words, n=1, cutoff=threshold)
        if matches:
            return True
    return False

def encode_image_to_base64_from_frame(frame):
    success, buffer = cv2.imencode('.jpg', frame)  # 转为 JPEG 格式的字节数据
    if not success:
        raise RuntimeError("图像编码失败")
    return base64.b64encode(buffer).decode("utf-8")

class ModelHistory:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}

class August(Node):
    def __init__(self):
        super().__init__('August_Cerebra')
        self.if_image = False
        self.target_node_name = '/signals_saver'
        self.client_set = self.create_client(SetParameters, self.target_node_name + '/set_parameters')
        self.client_get = self.create_client(GetParameters, self.target_node_name + '/get_parameters')
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50
        )
        self.cleaned_message_publisher_ = self.create_publisher(CleanedMessage, 'ai_answer', qos_profile)
        self.bridge = CvBridge()
        self.color_sub_ = self.create_subscription(
                            Image,
                            '/camera/color/image_raw',
                            self.listener_callback,
                            QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50
        ))
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.current_pose = None        # 记录当下的位置
        self.image_buffer_ = None


    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    # 计算两点之间的距离
    def calculate_distance(self, pose1, pose2):
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return math.sqrt(dx*dx + dy*dy)

    
    def listener_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Cannot convert image: {e}")
            return
        
        self.image_buffer_ = frame      # 不停的帮助存储图像，需要直接用就好了

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


    def parse_action_from_response(self, text: str):
        """
        从大模型输出中解析机器人动作指令（支持解析多个独立JSON对象）
        """
        actions = []
        try:
            # 使用 finditer 而不是 search，并采用非贪婪匹配 .*?
            # finditer 会返回一个迭代器，包含所有匹配的对象
            matches = re.finditer(r'\{.*?\}', text, re.DOTALL)
            
            if not matches:
                self.get_logger().warn("未找到任何有效的JSON动作指令。")
                return [] 

            for match in matches:
                try:
                    # 对每个匹配到的JSON字符串进行解析
                    action_json = loads(match.group())
                    actions.append(action_json)
                except json.JSONDecodeError as e:
                    # 即使找到了花括号，内部也可能不是合法的JSON
                    self.get_logger().warn(f"解析单个JSON动作失败: '{match.group()}', 错误: {e}")
                    continue # 跳过这个无效的，继续解析下一个

            # 如果成功解析了至少一个动作，则返回列表
            if actions:
                return actions
            else:
                # 所有找到的{}都是无效的JSON
                return []

        except Exception as e:
            # 捕获其他可能的意外错误
            self.get_logger().warn(f"动作解析过程中发生未知错误: {e}")
            return []
    
    def angle_diff(self, a, b):
        """
        计算角度 a 和 b 的差值，范围在 [-pi, pi]
        """
        diff = a - b
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return abs(diff)

    # 根据解析出的指令改变速度指令数值
    def execute_distance_action(self, actions: list):
        if actions:
            for action in actions:
                while self.current_pose is None:
                    self.get_logger().warn("尚未获取到里程计数据，无法执行动作")

                distance = float(action.get("value", 0.0))  # 目标距离/角度
                speed = 0.2  # 设定运动速度 (m/s)

                # 起始位置
                start_pose = self.current_pose

                # start_angle = quaternion_to_euler(start_pose.orientation)   

                if_turn = False     #判断是否转向

                msg = Twist()

                if action["action"] == "forward":
                    msg.linear.x = speed
                elif action["action"] == "backward":
                    msg.linear.x = -speed
                elif action["action"] == "left":
                    msg.linear.y = speed
                elif action["action"] == "right":
                    msg.linear.y = -speed
                elif action["action"] == "turn_left":
                    msg.angular.z = 0.87
                    if_turn = True
                elif action["action"] == "turn_right":
                    msg.angular.z = -0.87
                    if_turn = True
                else:
                    self.get_logger().warn(f"不支持的动作: {action['action']}")
                    return
                
                self.cmd_vel_pub.publish(msg)

                last_yaw = quaternion_to_euler(start_pose.orientation)[2]
                current_yaw = last_yaw
                accumulated_yaw = 0.0
                if_first_0 = True
                if_first_1 = True

                # 持续运动直到达到目标距离
                self.get_logger().info(f"开始执行动作: {action}")
                while rclpy.ok():
                    if not if_turn:
                        if self.current_pose is None:
                            self.get_logger().warn(f"Current-Pose retrieveing FAILED!!!")
                            continue

                        moved = self.calculate_distance(self.current_pose, start_pose)
                        if moved >= abs(distance):
                            break

                    else:
                        target_angle = math.radians(distance)   # 目标角度（弧度）

                        if self.current_pose is None:
                            self.get_logger().warn(f"Current-Pose retrieveing FAILED!!!")
                            continue

                        current_yaw = quaternion_to_euler(self.current_pose.orientation)[2]
                        turned = abs(self.angle_diff(current_yaw, last_yaw))

                        accumulated_yaw += turned

                        last_yaw = current_yaw

                        self.get_logger().info(
                            f"start_yaw: {quaternion_to_euler(start_pose.orientation)[2]}, current_yaw: {current_yaw}, accumulated: {accumulated_yaw}, target: {target_angle}"
                        )

                        if (target_angle - accumulated_yaw) < (40 * math.pi / 180) and if_first_0:
                            msg.linear.x = 0.0
                            msg.linear.y = 0.0
                            msg.angular.z = 0.25
                            self.cmd_vel_pub.publish(msg)
                            self.get_logger().info("距离目标角度小于40度，速度降为0.3m/rads")
                            if_first_0 = False
                        elif (target_angle - accumulated_yaw) < (20 * math.pi / 180) and if_first_1:
                            msg.linear.x = 0.0
                            msg.linear.y = 0.0
                            msg.angular.z = 0.05
                            self.cmd_vel_pub.publish(msg)
                            self.get_logger().info("距离目标角度小于20度，速度降为0.05m/rads")
                            if_first_1 = False
                        elif (target_angle - accumulated_yaw) <= (10 * math.pi / 180):
                            msg.linear.x = 0.0
                            msg.linear.y = 0.0
                            msg.angular.z = 0.0
                            self.cmd_vel_pub.publish(msg)
                            break

                    rclpy.spin_once(self, timeout_sec=0.01)  # 控制循环频率

                # 到达目标后停止
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.angular.z = 0.0
                self.cmd_vel_pub.publish(msg)
                self.get_logger().info("动作完成，机器人已停止")
        else:
            return

    def clean_for_tts(self, text: str) -> str:
        """
        去掉 JSON 符号和其它特殊字符
        """
        # 保留的字符集：中文、英文、数字、常用标点
        pattern = r'[^\u4e00-\u9fffA-Za-z0-9，,。.!！？?:：；;*#()（）\[\]【】{} ]'
        cleaned_text = re.sub(pattern, '', text)
        return cleaned_text

    # 用于监听终端指令
    def interrupt_listener(self):
        recognizer = hello_sherpa_onnx.initialize_recognizer(model_dir=model_dir)
        while True:
            recognized_text = hello_sherpa_onnx.speech_recognition(recognizer, if_interrupt=True)
            print("中断程序监听：{}".format(recognized_text))
            # keywords = {"停止朗读"}
            # if keywords.issubset(recognized_text.split()):
            if "停止朗读" in recognized_text:
                print("监听到中断！")
                tts_stop_event.set()
                self.try_set_param("tts_signal", True)

    def start_model(self, user_question: str):
        content = read_file_content(FILE_PATH)
        # 文件内容非空，则将其解析为 JSON 格式并赋值给 model_history_list；如果为空，则初始化为空列表
        model_history_list = loads(content) if content else []

        messages = []
        # 如果 model_history_list 不为空，则遍历列表，将每个历史记录添加到 messages 列表中；如果为空，则重新初始化为空列表
        if model_history_list:
            for history in model_history_list:
                messages.append({"role": history["role"], "content": history["content"]})
        else:
            model_history_list = []

        messages.append({"role": "system",
                        "content": """你的名字是：August。除非特别要求，你都要用中文回答问题。当你自我介绍时，你应该回答：“我是一款由Gerry Liu基于开源大模型进行二次开发，用于部署在RoboNova机器人上的可交互式智能体，简而言之我就是RoboNova的大脑。”；
                                        并且之后如果用户的问题中包含类似"august", "奥格斯", "奥格斯特", "欧格斯", "欧格斯特", "ogost", "ogs", "orgost", "OH GIR"的单词，你应该知道那不是问题的一部分，而是在叫你的名字；
                                        如果用户的问题涉及到机器人动作，请直接按照以下格式给出 JSON：
                                        {"action": "forward", "value": 1.0}
                                        其中 action 可取 "forward", "backward", "left", "right", "turn_left", "turn_right"，
                                        并且"left", "right"代表向左和向右移动，"turn_left", "turn_right" 代表左转和右转，value表示速度或角度大小。
                                        """})

        uq = user_question.lower()

        self.if_image = False
        if any(kw in uq for kw in ["描述", "告诉我", "讲一讲", "讲一下"]) and any(kw in uq for kw in ["眼前", "看到", "摄像头", "所看到"]):
            frame = self.image_buffer_
            # cv2.imshow("FRAME_FOR_AI", frame)
            # cv2.waitKey(1)

            image_base64 = encode_image_to_base64_from_frame(frame)

            messages.append({"role": "user", "content": user_question})
            print(f"MESSAGES: {messages}")
            self.if_image = True
        else:
            # 将当前用户的问题添加到 messages 列表中
            messages.append({"role": "user", "content": user_question})
            print(f"MESSAGES: {messages}")

        # 创建一个 ModelHistory 对象，表示用户的问题，并将其转换为字典格式后添加到 model_history_list 中
        model_history_list.append(ModelHistory("user", user_question).to_dict())

        # 构建请求体
        if self.if_image:
            print("Payload loaded!!!")
            payload = {
                "model": "llava",
                "prompt": "请用中文描述该图片中的内容",
                "images": [image_base64]
            }
        else:
            request_body = {
                "messages": messages,
                "model": "qwen3:8b",
                "stream": True,
                "temperature": 0.0,
            }

        # 构造请求头
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        if self.if_image:
            print("Request sent!!!")
            response = requests.post("http://localhost:11434/api/generate", json=payload)
        else:
            print("Request sent!!!")
            # 发送 POST 请求到模型的 API，获取响应
            response = requests.post("http://localhost:11434/v1/chat/completions", json=request_body, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Unexpected code {response.status_code}")

        msg = CleanedMessage()

        in_think_block = False   # 标记是否在 <think> 区间内

        # 初始化两个字符串 temp_res 和 out_res，用于存储临时和最终的响应内容
        temp_res = ""
        out_res = ""
        buffer = ""

        if_publish = False
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')         # 如果行不为空，则解码为 UTF-8 格式

                if self.if_image :
                    obj = loads(decoded_line)
                    chunk = obj.get("response", "")

                    temp_res += chunk
                    out_res += chunk
                    buffer += chunk

                    punctuations = "，,。.!！？?:：；;*#()（）[]【】{}"
                    if any(p in content for p in punctuations) and len(buffer) >= 30:
                        cleaned_string = buffer.replace("-", "")
                        msg.cleaned_message = cleaned_string
                        self.cleaned_message_publisher_.publish(msg)
                        # time.sleep(0.03)
                        print("buffer:{}".format(buffer))
                        buffer = ""
                        # if if_first:
                        #     self.try_set_param("consumer_signal", False)
                        #     if_first = False

                else:
                    if decoded_line.startswith("data: ") and "[DONE]" not in decoded_line:    # 如果行以 "data: " 开头且不包含 “[DONE]”，则处理该行
                        json_data = loads(decoded_line.replace("data: ", ""))      # 去除 "data: " 前缀，解析剩余部分为 JSON 数据
                        for choice in json_data["choices"]:
                            content = choice["delta"]["content"]

                            if not content:   # 有些块可能没有 content
                                continue

                            # 进入思考模式
                            if "<think>" in content:
                                in_think_block = True
                                continue
                            # 退出思考模式
                            if "</think>" in content:
                                in_think_block = False
                                continue
                            # 如果在 <think> 区间内，则跳过
                            if in_think_block:
                                continue

                            if tts_stop_event.is_set():
                                print("main_task::检测到中断，停止合成！")
                                return

                            temp_res += content
                            out_res += content      #out_res 中的是完整的回答

                            buffer += self.clean_for_tts(content)
                            punctuations = "，,。.!！？?:：；;*#()（）[]【】{}"
                            if any(p in content for p in punctuations) and len(buffer) >= 30:
                                cleaned_string = buffer.replace("-", "")
                                msg.cleaned_message = cleaned_string
                                self.cleaned_message_publisher_.publish(msg)
                                if_publish = True
                                time.sleep(0.03)
                                self.get_logger().info("MESSAGE PUBLISHED 2222!!!")
                                print("buffer:{}".format(buffer))
                                buffer = ""

                                # if if_first:
                                #     self.try_set_param("consumer_signal", False)
                                #     if_first = False

        self.try_set_param("consumer_signal", False)        # 保险起见

        if not self.if_image:
            # 使用正则表达式去除 out_res 中的 <think> 标签及其内容
            out_res = re.sub(r'<think>[\s\S]*?</think>', '', out_res)

        # 尝试解析并执行机器人动作
        action = self.parse_action_from_response(out_res)
        if action:
            # self.execute_distance_action(action)
            thread = threading.Thread(target=self.execute_distance_action, args=(action,))
            thread.start()

        print(f"Length of out_res: {len(out_res)}")
        if not if_publish:
            msg1 = CleanedMessage()
            msg1.cleaned_message = out_res
            msg1.cleaned_message = self.clean_for_tts(msg1.cleaned_message)
            self.cleaned_message_publisher_.publish(msg1)
            self.get_logger().info("MESSAGE PUBLISHED 333!!!")

            # time.sleep(3)
            # self.try_set_param("consumer_signal", True)
            # self.get_logger().warn(f"已自动将consumer_signal参数更改为TRUE！！！")

        # 创建一个 ModelHistory 对象，表示模型的回答，并将其转换为字典格式后添加到 model_history_list 中
        model_history_list.append(ModelHistory("assistant", out_res).to_dict())

        write_file_content(FILE_PATH, dumps(model_history_list, indent=4, ensure_ascii=False))

    def remove_wake_word_regex(self, text: str, wake_words: list) -> str:
        """
        使用正则表达式从文本中移除唤醒词。
        """
        # 将唤醒词列表中的特殊字符进行转义，并用 | 连接，形成一个 "word1|word2|..." 的模式
        # re.escape 确保像 '(' 或 '.' 这样的字符被当作普通字符处理
        pattern = '|'.join(map(re.escape, wake_words))
        
        # re.IGNORECASE 标志使匹配不区分大小写
        # re.sub 会查找所有匹配的子串并用空字符串替换
        processed_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 同样，清理多余的空格
        processed_text = processed_text.strip()
        processed_text = ' '.join(processed_text.split())
        
        return processed_text

    def start_iat(self, model_dir=model_dir):

        playsound("/home/gerry/下载/RA2 xa4ei02.mp3")
        print("开始调用听写")
        recognizer = hello_sherpa_onnx.initialize_recognizer(model_dir)
        try:
            while True:
                tts_stop_event.clear()
                self.try_set_param("tts_signal", False)

                print("等待音频输入..................")

                recognized_text = hello_sherpa_onnx.speech_recognition(recognizer)
                self.try_set_param("consumer_signal", False)

                # 开启监听中断指令线程
                listener_thread = Thread(target=self.interrupt_listener, daemon=True)
                listener_thread.start()

                wake_words = ["august", "奥格斯", "奥格斯特", "欧格斯", "欧格斯特", "ogost", "ogs", "orgost", "OH GIR"]
                if is_wake_word_in_sentence(recognized_text, wake_words):
                    recognized_text = self.remove_wake_word_regex(recognized_text, wake_words)
                    self.start_model(recognized_text)
                    print("模型回答中..................")
                else:
                    self.try_set_param("consumer_signal", True)

                consumer_signal_status = self.try_get_param("consumer_signal")
                self.get_logger().info(f"{consumer_signal_status}  111111111111111111111111111111111111111111111")
                while not consumer_signal_status:
                    time.sleep(.2)
                    if consumer_signal_status:
                        print("consumer_signal has turned TRUE!!!")
                    consumer_signal_status = self.try_get_param("consumer_signal")

        except Exception as e:
            print(e)

    def img_show_fuc(self):
        while True:
            if self.if_image:
                cv2.imshow("IMAGE_FOR_AI", self.image_buffer_)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                    


def main():
    rclpy.init()
    august_node = August()
    iat_thread = Thread(target=august_node.start_iat)
    iat_thread.start()
    image_thread = Thread(target=august_node.img_show_fuc)
    image_thread.start()
    rclpy.spin(august_node)
    rclpy.shutdown()


