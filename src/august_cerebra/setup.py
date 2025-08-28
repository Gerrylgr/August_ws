from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'august_cerebra'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'knowledge_history'), glob(os.path.join('knowledge_history', '*.*'))),
        (os.path.join('share', package_name, 'models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20'), glob(os.path.join('models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20', '*.*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gerry',
    maintainer_email='2717915639@qq.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main_task_ros = august_cerebra.main_task_ros:main',
        ],
    },
)
