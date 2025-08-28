# 应用于ROS机器人的智能体框架August

## 1.项目介绍

本项目可将ros机器人接入智能体（名为August），完全在本地实现语音交互、图片识别、运动控制、构建本地知识库等功能。

各功能包描述如下：
- august_cerebra:TTS引擎和调用ollama接口获取模型回答的主要文件，在电脑端运行
- august_speaker:STT引擎，用于将模型回答转为语音的主要文件，在机器人主控端运行
- signals_node：存放标志位的节点，主要用于处理标志位的相关参数请求，推荐运行在电脑端
- cleaned_message_interface:用于存储模型回答的自定义消息，电脑端和主控端都需要有

## 2.使用方法

本项目开发平台信息如下：

- 系统版本： Ubunt22.04
- ROS 版本：ROS 2 Humble

### 2.1安装

本项目由STT引擎、ollama模型和TTS引擎三大部分构成，构建之前请先安装依赖(默认已安装好ROS2 humble 并配置好电脑和机器人主控的多机通讯)：

1. 安装 STT 引擎（sherpa-onnx）相关依赖：

    见 https://github.com/k2-fsa/sherpa-onnx 并按照官方指示安装依赖；

    验证是否安装成功：
    (找不到包的话请把 main 中模型的路径改为绝对路径)

    ```
    python3 src/august_cerebra/august_cerebra/hello_sherpa_onnx.py 
    ```

    若能够识别语音输入则证明依赖安装完成

2. 安装 TTS 引擎（MeloTTS）相关依赖：

    见 https://github.com/myshell-ai/MeloTTS 并按照官方指示安装依赖；

    验证是否安装成功：
    (找不到包的话请把 main 中模型的路径改为绝对路径)

        ```
        python3 src/august_speaker/august_speaker/MeloTest.py 
        ```

        若能够成功合成语音则说明依赖安装完成

3. 安装 ollama 并拉取模型到本地：

    见 https://ollama.com/download 并下载ollama

    拉取模型（此处示例使用代码中的 qwen3:8b 和 llava:latest）：

    ```
    ollama serve
    ollama run qwen3:8b
    ollama run llava:latest
    ```

### 2.2运行

安装完成依赖后，使用 colcon 工具进行构建和运行。

构建功能包

```
colcon build
```

运行参数节点：

```
source install/setup.bash
ros2 run signals_node signals_saver 
```

运行 TTS 节点（机器人主控端）：

```
source install/setup.bash
ros2 run august_speaker MeloSpeaker
```

运行大模型和 TTS 节点：（电脑端）

```
source install/setup.bash
ros2 run august_cerebra main_task_ros 
```

### 2.3注意

因模型本体较大，不直接存放在github仓库中，可对照.gitignore文件，前往官网自行下载对应的模型文件；

（附sherpa-onnx模型下载链接：
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english

MeloTTS 模型安装请前往官方仓库 https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md 按照说明安装）

运行该智能体时请配置好电脑和机器人主控的多机通讯；

该智能体框架图片识别接口默认订阅 sensor_msgs/msg/Image 类型的 /camera/color/image_raw 话题，如果不同请自行修改；

默认的语音交互模型为 qwen3:8b，图片识别模型为 llava，如有需要可在 main_task_ros.py 中修改；

## 3.作者

- [Gerry Liu](https://github.com/Gerrylgr?tab=repositories)