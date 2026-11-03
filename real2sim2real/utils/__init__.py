"""辅助工具(位姿、落盘、Isaac 舞台).平台核心在 ``real2sim2real.core``.

``isaacsim_utils`` 子包按类型拆分(openusd / modify_stage / physics / camera),包入口导出常用符号;
``pose`` 做齐次变换与 ``PoseConfig``;
``recorder`` / ``validate`` 做 LeRobot 落盘与事后校验.
"""
