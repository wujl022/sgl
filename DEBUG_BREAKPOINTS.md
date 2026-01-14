# 调试断点位置指南

## 使用 VS Code 调试器

### 1. 启动调试
- 按 `F5` 或点击"运行和调试"
- 选择 "Debug Mini-SGLang Server" 配置
- 程序会以调试模式启动

### 2. 设置断点的关键位置

## 📍 模型部署流程的关键断点位置

### 阶段1：服务器启动和参数解析

**文件：`python/minisgl/server/launch.py`**
- **第 44 行**：`server_args, run_shell = parse_args(sys.argv[1:], run_shell)`
  - 观察：参数是否正确解析，模型路径是否正确

**文件：`python/minisgl/server/launch.py`**
- **第 21 行**：`scheduler = Scheduler(args)`
  - 观察：Scheduler 初始化，这是模型加载的入口

---

### 阶段2：Engine 初始化和模型创建（⭐ 最重要）

**文件：`python/minisgl/engine/engine.py`**

#### 2.1 模型结构创建
- **第 55 行**：`self.model = create_model(config.model_path, config.model_config)`
  - 观察：模型结构是否创建成功
  - 检查：`self.model` 的结构

#### 2.2 权重加载
- **第 56 行**：`self.model.load_state_dict(self._load_weight_state_dict(config))`
  - 观察：权重是否正确加载
  - 检查：加载的权重字典

#### 2.3 权重后处理（⭐ 关键）
- **第 57 行**：`self.model.process_weights_after_loading()`
  - 观察：权重处理流程，包括 GPTQ 量化处理
  - **这是最重要的断点位置！**

---

### 阶段3：Linear 层权重处理（GPTQ 量化）

**文件：`python/minisgl/layers/linear.py`**

#### 3.1 权重处理入口
- **第 161 行**：`def process_weights_after_loading(self) -> None:`
  - 观察：每个 Linear 层开始处理权重
  - 检查：`self._gptq` 配置

#### 3.2 GPTQ 配置检查
- **第 162 行**：`if self._gptq is None:`
  - 观察：是否有 GPTQ 配置

#### 3.3 Marlin 格式检查
- **第 173 行**：`if self._use_marlin:`
  - 观察：是否已经使用 Marlin 格式

#### 3.4 Marlin 格式转换（⭐ 你关注的代码）
- **第 191 行**：`if not self._gptq.is_marlin_format:`
  - 观察：非 Marlin 格式转换为 Marlin 格式的过程
  - 检查：
    - `self._gptq.desc_act` 的值
    - `self.g_idx` 的形状和内容
    - `self.qweight` 的形状

#### 3.5 desc_act 处理
- **第 192 行**：`if self._gptq.desc_act:`
  - 观察：desc_act=True 时的处理逻辑

#### 3.6 权重重排
- **第 200 行**：`self.qweight = gptq_marlin_repack(...)`
  - 观察：权重重排操作
  - 检查：重排前后的权重

#### 3.7 Scales 重排
- **第 207 行**：`self.scales = _marlin_permute_scales(...)`
  - 观察：scales 的重排操作

#### 3.8 Qzeros 处理
- **第 213 行**：`self.qzeros = _marlin_make_empty_int(device)`
  - 观察：qzeros 的处理

---

### 阶段4：KV Cache 初始化

**文件：`python/minisgl/engine/engine.py`**
- **第 59 行**：`self.kv_cache = create_kvcache(...)`
  - 观察：KV cache 的创建

**文件：`python/minisgl/engine/engine.py`**
- **第 67 行**：`self.page_table = create_page_table(...)`
  - 观察：页面表的创建

---

### 阶段5：注意力后端初始化

**文件：`python/minisgl/engine/engine.py`**
- **第 71 行**：`self.attn_backend = create_attention_backend(...)`
  - 观察：注意力后端的创建

---

### 阶段6：Scheduler 运行

**文件：`python/minisgl/scheduler/scheduler.py`**
- **第 48 行**：`self.engine = Engine(config)`
  - 观察：Engine 创建（会触发上面的所有初始化）

**文件：`python/minisgl/scheduler/scheduler.py`**
- **第 31 行**：`scheduler.run_forever()`
  - 观察：调度器开始运行

---

## 🎯 推荐的断点设置顺序

### 第一次调试（了解整体流程）：
1. `engine.py:57` - 权重后处理入口
2. `linear.py:161` - Linear 层权重处理入口
3. `linear.py:191` - Marlin 格式转换

### 深入调试 GPTQ 处理：
1. `linear.py:161` - 每个 Linear 层
2. `linear.py:191` - Marlin 格式转换
3. `linear.py:192` - desc_act 判断
4. `linear.py:200` - 权重重排
5. `linear.py:207` - Scales 重排
6. `linear.py:213` - Qzeros 处理

---

## 💡 调试技巧

### 1. 条件断点
在 VS Code 中，右键断点可以设置条件，例如：
- `self._gptq is not None` - 只在有 GPTQ 配置时停止
- `self._gptq.desc_act == True` - 只在 desc_act=True 时停止
- `"qkv" in str(type(self).__name__)` - 只在特定层停止

### 2. 日志断点
在断点处自动打印变量值，而不停止执行

### 3. 观察变量
在"监视"面板添加：
- `self._gptq`
- `self.qweight.shape`
- `self.g_idx`
- `self.scales.shape`

### 4. 调用堆栈
查看调用堆栈了解代码执行路径

---

## 📝 调试检查清单

在断点处检查：

1. **模型加载阶段**：
   - [ ] 模型路径是否正确
   - [ ] 模型结构是否正确创建
   - [ ] 权重是否正确加载

2. **GPTQ 处理阶段**：
   - [ ] `self._gptq` 配置是否正确
   - [ ] `self._gptq.bits` 是否为 4 或 8
   - [ ] `self._gptq.group_size` 是否正确
   - [ ] `self._gptq.desc_act` 的值
   - [ ] `self._gptq.is_marlin_format` 的值

3. **权重处理阶段**：
   - [ ] `self.qweight` 的形状是否正确
   - [ ] `self.scales` 的形状是否正确
   - [ ] `self.g_idx` 的内容（如果 desc_act=True）
   - [ ] `self.qzeros` 的处理是否正确

---

## 🚀 快速开始

1. 在 VS Code 中打开项目
2. 按 `F5` 启动调试
3. 在推荐的位置设置断点
4. 观察变量和调用堆栈
5. 使用 `F10` (单步跳过) 和 `F11` (单步进入) 逐步执行

