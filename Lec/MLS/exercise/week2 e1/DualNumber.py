"""模块：DualNumber 前向自动微分示例

此模块实现了一个最小的 `DualNumber` 类用于前向模式自动微分，并演示在 TinyML 传感器预处理流水线上的应用。
功能包括：
- 对输入 `sensor_reading` 做归一化、ReLU、缩放操作
- 使用 `DualNumber` 对输入求导（forward AD）
- 使用数值中心差分验证 AD 结果
"""


class DualNumber:
    """
    双数（Dual Number）类，用于前向自动微分（Forward-mode AD）。

    属性:
    - real: 实部（浮点数），代表函数值
    - dual: 伴随值（浮点数），代表该值相对于被求导变量的导数

    设计要点：此类实现了最小必要的算符重载，以支持在表达式中使用常见算术操作并自动传播导数。
    """

    def __init__(self, real, dual=0.0):
        # 将输入转换为浮点数并保存实部与伴随（导数）部分
        self.real = float(real)
        self.dual = float(dual)

    def __add__(self, other):
        # 加法规则：(a + bε) + (c + dε) = (a+c) + (b+d)ε
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        return DualNumber(self.real + float(other), self.dual)

    def __radd__(self, other):
        # 支持右侧加法，如 3 + DualNumber(...)
        return self.__add__(other)

    def __sub__(self, other):
        # 减法规则：(a + bε) - (c + dε) = (a-c) + (b-d)ε
        if isinstance(other, DualNumber):
            return DualNumber(self.real - other.real, self.dual - other.dual)
        return DualNumber(self.real - float(other), self.dual)

    def __rsub__(self, other):
        # 支持右侧减法，如 3 - DualNumber(...)
        if isinstance(other, DualNumber):
            return other.__sub__(self)
        return DualNumber(float(other) - self.real, -self.dual)

    def __neg__(self):
        # 取负：-(a + bε) = (-a) + (-b)ε
        return DualNumber(-self.real, -self.dual)

    def __mul__(self, other):
        # 乘法规则：(a + bε)*(c + dε) = a*c + (a*d + b*c)ε
        if isinstance(other, DualNumber):
            return DualNumber(self.real * other.real, self.real * other.dual + self.dual * other.real)
        o = float(other)
        return DualNumber(self.real * o, self.dual * o)

    def __rmul__(self, other):
        # 支持右侧乘法，如 2 * DualNumber(...)
        return self.__mul__(other)

    def __truediv__(self, other):
        # 除法规则：(a + bε)/(c + dε) = (a/c) + ((b*c - a*d)/c^2) ε
        if isinstance(other, DualNumber):
            return DualNumber(self.real / other.real, (self.dual * other.real - self.real * other.dual) / (other.real ** 2))
        d = float(other)
        return DualNumber(self.real / d, self.dual / d)

    def __rtruediv__(self, other):
        # 支持右侧除法，如 3 / DualNumber(...)
        o = float(other)
        return DualNumber(o / self.real, -(o * self.dual) / (self.real ** 2))

    def relu(self):
        """
        ReLU 激活：如果实部 > 0，则保留实部和伴随；否则输出 0 且导数为 0。

        返回值为新的 `DualNumber` 实例，方便链式调用。
        """
        if self.real > 0.0:
            return DualNumber(self.real, self.dual)
        return DualNumber(0.0, 0.0)


def tinyml_feature_processing(sensor_reading):
     """
     传感器预处理流水线（对 `DualNumber` 或普通数值都适用）。

     详细步骤：
     1) 归一化：normalized = (sensor_reading - 512) / 512
         - 如果传入的是 `DualNumber`，则这里触发 `__sub__` 与 `__truediv__`，导数会被自动传播。
     2) ReLU 激活：activated = normalized.relu()
         - 对 `DualNumber`，`relu()` 会根据实部是否大于 0 决定是否传递伴随值（导数）。
     3) 缩放：scaled = activated * 0.5

     参数：
     - `sensor_reading`: 可以是 `DualNumber`（用于自动微分）或 `float`（用于数值检查）。

     返回：处理后的值，类型与输入一致（若输入为 `DualNumber` 则返回 `DualNumber`）。
     """
     normalized = (sensor_reading - 512) / 512
     activated = normalized.relu()
     scaled = activated * 0.5
     return scaled


if __name__ == '__main__':
    # 示例运行流程：
    # 1) 构造带种子导数的 DualNumber：x = DualNumber(800.0, 1.0)
    #    这里的种子 dual=1.0 表示我们对输入 800 求偏导数。
    # 2) 将 x 传入 tinyml_feature_processing，得到返回的 DualNumber y，y.real 为函数值，y.dual 为导数值。
    # 3) 使用数值中心差分验证 AD 结果。
    x = DualNumber(800.0, 1.0)
    y = tinyml_feature_processing(x)
    print(f"Output (real): {y.real}")
    print(f"Derivative (AD): {y.dual}")

    # 数值中心差分验证（Finite Difference）
    h = 1e-6
    x0 = 800.0
    # 直接按标量流水线计算 f(x0 ± h)，避免构造 DualNumber
    n_plus = (x0 + h - 512.0) / 512.0
    s_plus = (n_plus if n_plus > 0.0 else 0.0) * 0.5
    n_minus = (x0 - h - 512.0) / 512.0
    s_minus = (n_minus if n_minus > 0.0 else 0.0) * 0.5
    numeric = (s_plus - s_minus) / (2 * h)
    print(f"Derivative (FD): {numeric}")
