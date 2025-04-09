# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time  # 导入时间模块，用于获取当前时间或生成时间戳
from dataclasses import dataclass, field, fields, is_dataclass  # 导入数据类相关工具
# dataclass: 用于定义数据类，自动生成__init__、__repr__等方法
# field: 用于自定义数据类字段的默认值或行为
# fields: 用于获取数据类的字段信息
# is_dataclass: 用于检查一个对象是否是数据类
from enum import StrEnum  # 导入字符串枚举类，用于定义枚举类型，枚举值会自动转换为字符串
from typing import Any, Self, TypeVar  # 导入类型注解相关类型
# Any: 表示任意类型，用于类型注解中表示可以是任何类型
# Self: 表示当前类的类型，通常用于类方法的返回类型注解
# TypeVar: 用于定义泛型类型变量，支持泛型编程

T = TypeVar("T") # 定义一个泛型类型变量 T。T 是一个类型变量：这里的 T 是一个占位符，表示一个未知的类型。它可以是任何类型，比如 int、str、list 等。
# 在 config.py 中，T 被用于 from_dict 函数的类型注解中

def from_dict(cls: type[T], dikt: dict[str, Any]) -> T:
    """Restore dataclass from a OmegaConf dictionary.

    Args:
        cls: The class of the dataclass
        dikt: The dictionary containing the properties of the dataclass

    Raises:
        ValueError: the dictionary and the dataclass is not compatible

    Returns:
        The dataclass instance.
    """
    try:
        fieldtypes = {f.name: f.type for f in fields(cls)}  # type: ignore 
        # type: ignore 不仅仅是一个注释，它在类型检查工具中具有实际功能，用于忽略特定行的类型检查错误。虽然它看起来像注释，但它对类型检查器的行为有直接影响。
        # 获取数据类 cls 的所有字段，并创建一个字典，键为字段名，值为字段类型
        # 使用 fields(cls) 获取 cls 的所有字段信息
        # 使用 f.name 获取字段的名称
        # 使用 f.type 获取字段的类型
        # 使用 # type: ignore 忽略类型检查错误
        return cls(
            **{
                f: from_dict(fieldtypes[f], dikt[f])  # type: ignore
                # 如果字段类型是一个数据类，则递归调用 from_dict 函数进行处理
                # 否则直接使用字典中的值
                # 使用 is_dataclass(fieldtypes[f]) 检查字段类型是否是一个数据类
                # 如果字段类型是一个数据类，则递归调用 from_dict 函数进行处理                
                if is_dataclass(fieldtypes[f])
                # 否则直接使用字典中的值
                else dikt[f]
                for f in dikt
                if f in fieldtypes  # allow extra keys
            }
        )
    except Exception as e:
        raise ValueError(f"Error converting dictionary to {cls.__name__}: {e}")


class InteractionType(StrEnum): # (StrEnum)：继承自 StrEnum，表示这是一个字符串枚举类。
    coulomb = "coulomb" #coulomb：枚举成员，表示库仑相互作用。"coulomb"：枚举成员的值，是一个字符串。
    harmonic = "harmonic"
# 与普通的 Enum 不同，StrEnum 的成员可以直接作为字符串使用，而不需要调用 .value。
# 访问枚举成员, print(InteractionType.coulomb)  # 输出: coulomb


@dataclass
class System:
    flux: int = 2
    "Positive or negative integer $2Q$."

    radius: float | None = None
    r"By default, the radius of the sphere is fixed at $\sqrt{Q}$."
    # r"..."：这是一个原始字符串（raw string），用于表示文档字符串（docstring）。使用 r 前缀可以避免字符串中的反斜杠 \ 被解释为转义字符。

    nspins: tuple[int, int] = (3, 0)
    "Number of spin-up and spin-down electrons."

    interaction_strength: float = 1.0
    "The factor for the potential energy."

    lz_center: float = 0.0
    "Lz to pick using penalty method."

    lz_penalty: float = 0.0
    "The strength of the penalty for (Lz - lz_center)^2."

    l2_penalty: float = 0.0
    "The strength of the penalty for L^2."

    interaction_type: InteractionType = InteractionType.coulomb


class NetworkType(StrEnum):
    psiformer = "psiformer"
    laughlin = "laughlin"
    free = "free"


class OrbitalType(StrEnum):
    full = "full"
    sparse = "sparse"


@dataclass
class PsiformerNetwork:
    num_heads: int = 4
    heads_dim: int = 64
    num_layers: int = 2
    determinants: int = 1


@dataclass
class Network:
    type: NetworkType = NetworkType.psiformer
    orbital: OrbitalType = OrbitalType.full
    psiformer: PsiformerNetwork = field(default_factory=PsiformerNetwork)
    # (字段名称：字段类型 = 默认值)
    # field 函数是 Python dataclasses 模块中的一个工具，用于自定义数据类字段的行为。它允许你更精细地控制字段的初始化、默认值、默认工厂函数等。
    # default_factory 是一个工厂函数，用于在创建数据类实例时动态生成默认值。
    # PsiformerNetwork：是 PsiformerNetwork 数据类的一个实例。
    # PsiformerNetwork 是工厂函数，表示每次创建数据类实例时，都会调用 PsiformerNetwork() 生成一个新的 PsiformerNetwork 实例。
    
    # default_factory：为字段指定一个工厂函数，用于在创建数据类实例时动态生成默认值。
    # from dataclasses import dataclass, field
    # @dataclass
    # class Example:
    #     items: list = field(default_factory=list)

    # default：为字段指定一个静态的默认值。
    #from dataclasses import dataclass, field
    # @dataclass
    # class Example:
    #     value: int = field(default=42)


@dataclass
class MCMC:
    steps: int = 10
    "MCMC steps to run between steps."

    width: float = 0.1
    "The std dev for gaussian move."

    burn_in: int = 200
    """MCMC burn in steps to run before training.

    It's actually `mcmc.burn_in * mcmc.steps` number of steps.
    """

    adapt_frequency: int = 100
    "Number of steps after which to update the adaptive MCMC step size."


@dataclass
class LearningRate:
    """Define the learning rate.

    The formula is rate * (1.0 / (1.0 + (t / delay)) ** decay
    """

    rate: float = 0.005
    decay: float = 1.0
    delay: float = 2000.0

    def schedule(self, t):
        return self.rate * (1.0 / (1.0 + (t / self.delay))) ** self.decay


class OptimizerName(StrEnum):
    adam = "adam"
    kfac = "kfac"
    none = "none"


@dataclass
class OptimizerAdam:
    lr: LearningRate = field(default_factory=LearningRate)


@dataclass
class OptimizerKfac:
    lr: LearningRate = field(default_factory=lambda: LearningRate(rate=0.05))


@dataclass
class Optim:
    iterations: int = 1000
    optimizer: OptimizerName | None = OptimizerName.kfac
    adam: OptimizerAdam = field(default_factory=OptimizerAdam)
    kfac: OptimizerKfac = field(default_factory=OptimizerKfac)


@dataclass
class Log:
    save_path: str | None = None
    """Path to save checkpoints and logs.

    Can be any path supported by fsspec/universal_pathlib.
    """

    restore_path: str | None = None
    """
    Path to restore checkpoints.

    Can be a directory containing checkpoints or path to a specific checkpoint.
    """

    save_time_interval: int = 10 * 60
    """Minimum time (in seconds) between checkpoint saves.

    A checkpoint will only be saved if both this interval has passed and
    the current step is a multiple of `save_step_interval`.
    """

    save_step_interval: int = 1000
    """Checkpoints are saved only at steps that are multiples of this value.

    Checkpoints are saved only at steps that are multiples of this value,
    and only if the `save_time_interval` has also elapsed.
    """

    initial_energy: bool = True
    """Log initial energy before any optimizations.

    This is helpful for debugging. If we have initial energy but have error in training,
    it's probably optimizer's fault
    """


@dataclass
class Config:
    batch_size: int = 3360  # 32*3*5*7
    seed: int = field(default_factory=lambda: int(time.time())) 
    # lambda 是 Python 中用于定义匿名函数的关键字。匿名函数是一种没有名称的函数，通常用于简化代码或作为参数传递给其他函数。
    # 基本语法：lambda 参数: 表达式。lambda 函数可以接受任意数量的参数，但只能有一个表达式。表达式的结果会自动返回，不需要 return 语句。
    system: System = field(default_factory=System)
    network: Network = field(default_factory=Network)
    mcmc: MCMC = field(default_factory=MCMC)
    optim: Optim = field(default_factory=Optim)
    log: Log = field(default_factory=Log)

    @classmethod
    def from_dict(cls, dikt: dict) -> Self: # 返回值：Self 表示返回类型是当前类的实例，即 Config 实例。
        # 装饰器：将 from_dict 方法声明为类方法。
        # 作用：类方法的第一个参数是类本身（通常命名为 cls），而不是实例。可以通过类直接调用该方法，而不需要创建实例。
        """Convert a dictionary to Config.""" # 方法的文档字符串，描述方法的作用。
        return from_dict(cls, dikt) # 调用全局的 from_dict 函数，将字典转换为 Config 实例。
    
    # 作用域不同：
    # def from_dict 是 Config 类中的方法，作用域在类内部。
    # return from_dict 调用的是全局的 from_dict 函数，作用域在模块级别。
    # Python 的作用域规则：
    # Python 允许在不同作用域中使用相同的名称，因为它们不会冲突。
    # 在 Config.from_dict 方法中，from_dict 首先查找局部作用域（类方法），如果没有找到，则查找全局作用域（模块级别的函数）。

    # 解释：
    # 1. `def from_dict` 是 `Config` 类中的一个类方法，用于将字典转换为 `Config` 实例。
    # 2. `return from_dict(cls, dikt)` 调用了全局的 `from_dict` 函数，而不是 `Config` 类中的 `from_dict` 方法。
    # 3. Python 的名称解析遵循 LEGB 规则（Local -> Enclosing -> Global -> Built-in）：
    #    - 首先查找局部作用域（`Config.from_dict` 方法内部），没有找到 `from_dict` 的定义。
    #    - 然后查找嵌套作用域（`Config` 类的作用域），也没有找到 `from_dict` 的定义。
    #    - 最后在全局作用域中找到了全局的 `from_dict` 函数，并调用它。
    # 4. `Config.from_dict` 是类方法，必须通过 `Config.from_dict(...)` 的形式调用，而 `from_dict(cls, dikt)` 是直接调用函数，因此不会解析为 `Config.from_dict` 方法。
    # 5. 这种行为是 Python 作用域规则和名称解析机制的结果，确保了代码的正确执行。
