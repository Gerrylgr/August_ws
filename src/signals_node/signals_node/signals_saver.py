import rclpy
from rclpy.node import Node

class TargetNode(Node):
    def __init__(self):
        super().__init__('signals_saver')

        self.declare_parameter('tts_signal', False)
        self.declare_parameter('consumer_signal', False)

        # 定时器：每 2 秒读取并打印参数值
        self.timer = self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        param = self.get_parameter('tts_signal').value
        self.get_logger().info(f'tts_signal = {param}')
        param = self.get_parameter('consumer_signal').value
        self.get_logger().info(f'consumer_signal = {param}')
        print("------------------------------")

def main(args=None):
    rclpy.init(args=args)
    node = TargetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()