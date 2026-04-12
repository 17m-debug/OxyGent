# OxyBank 使用说明

## 简介

OxyBank 是一个资产库管理与标注系统，帮助您轻松创建、管理和检索资产。通过简单直观的操作，您可以建立自己的资产库，上传文档，并快速检索所需信息，同时支持各资产进行标注，打通数据资产管理与标注全流程

![OxyBank系统概览](https://storage.jd.com/ai-gateway-routing/prod_data/oxybank/system-overview.png)

## 系统要求

- **Python**: 3.10 及以上版本
- **Node.js**: 16.0 及以上版本（用于前端）

## 安装步骤

### 1. 获取项目代码

```bash
cd applications/OxyBank
```

### 2. 安装后端依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 安装前端依赖

```bash
cd web
npm install
```

### 4. 配置环境变量

创建文件 `.env`，并根据您的环境修改配置：

```bash
APP_NAME=OxyBank

ELASTICSEARCH_HOSTS=
ELASTICSEARCH_USERNAME=
ELASTICSEARCH_PASSWORD=

VEARCH_HOST=
VEARCH_TOKEN=
VEARCH_DB_NAME=knowledge_base
VEARCH_VECTOR_DIMENSION=2048

# ==================== Embedding Configuration ====================
# Embedding type: glm
EMBEDDING_TYPE=glm
EMBEDDING_API_KEY=

# GLM Embedding Settings (when EMBEDDING_TYPE=glm)
EMBEDDING_GLM_MODEL_NAME=embedding-3
EMBEDDING_GLM_API_BASE=https://open.bigmodel.cn/api/paas/v4/embeddings

# ============= Annotation platform Configuration =============
# ES Index Settings
ANNOTATION_ES_INDEX_PREFIX=qa_annotation
ANNOTATION_ES_DEDUP_ENABLED=true

# KB Ingest Settings
ANNOTATION_KB_ENABLED=true
ANNOTATION_KB_ID=
ANNOTATION_KB_AUTO_INGEST=false
ANNOTATION_KB_TIMEOUT=30
ANNOTATION_KB_RETRY_TIMES=3
ANNOTATION_KB_RETRY_INTERVAL=5

# Data Check Settings
ANNOTATION_BATCH_SIZE=100

# Data Type Infer Settings
ANNOTATION_DEFAULT_DATA_TYPE=custom
ANNOTATION_DEFAULT_PRIORITY=4

```

## 启动服务

### 启动后端服务

```bash
# 在项目根目录
python app/main.py
```

成功启动后，您会看到类似以下的输出：
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 访问系统

1. 打开浏览器，访问前端地址：http://0.0.0.0:8000
2. 系统界面将显示资产库管理页面

## 使用指南

### 创建您的第一个资产库

#### 步骤1：进入资产库管理
- 点击左侧菜单中的"资产库"选项
- 您将看到资产库列表页面

![资产库列表界面](https://storage.jd.com/ai-gateway-routing/prod_data/oxybank/kb-list.png)

#### 步骤2：创建新资产库
- 点击"创建资产库"按钮
- 填写资产库基本信息：
  - **名称**：为您的资产库取一个易记的名字
  - **描述**：简要描述资产库的用途
  - **类型**：选择"结构化"或"非结构化"

![创建资产库界面](https://storage.jd.com/ai-gateway-routing/prod_data/oxybank/create-kb.png)

#### 步骤3：上传文档
- 根据资产库类型，上传相应的文档：
  - **结构化资产库**：上传Excel或CSV文件
  - **非结构化资产库**：上传TXT、MarkDown、html等文本文件
- 系统将自动处理您的文档

#### 步骤4：配置检索规则（可选）
- 对于结构化资产库，您可以配置检索字段和检索规则（全文关键词检索、向量语义检索）
- 对于非结构化资产库，您可以设置文本分段参数，当前默认配置全文关键词检索和向量语义检索两种检索策略


![索引策略配置](https://storage.jd.com/ai-gateway-routing/prod_data/oxybank/search.png)

#### 步骤5：完成创建
- 点击"创建"按钮完成资产库创建
- 系统将自动为您的资产库生成检索接口（根据检索策略生成相应数目检索接口）

### 管理资产库

#### 查看资产库
- 在资产库列表中，点击资产库名称查看详情
- 您可以查看资产库的基本信息、文档列表和检索召回接口

#### 添加新文档
- 在资产库详情页面，点击"上传文档"按钮
- 选择要上传的文件并确认

#### 删除资产库
- 在资产库列表中，点击资产库右侧的"删除"按钮
- 确认删除操作（注意：此操作不可恢复）

### 检索资产

#### 使用前端检索
- 在资产库详情页面，使用检索框输入关键词
- 系统将显示匹配的检索结果
- 您可以进一步筛选和查看详细内容

![检索界面示例](https://storage.jd.com/ai-gateway-routing/prod_data/oxybank/search-example.png)

#### 使用API检索
每个资产库都会自动生成检索接口，格式如下：

```
POST http://localhost:8000/kb/{资产库名称}/search/rule_0
```

示例请求：
```bash
curl -X POST http://localhost:8000/kb/my_bank/search/rule_0 \
  -H "Content-Type: application/json" \
  -d '{"query": "您的搜索关键词"}'
```

## 常见问题

### 1. 服务启动失败怎么办？

**问题**：运行 `python app/main.py` 时报错
**解决**：
- 检查Python版本是否为3.8+
- 确保所有依赖已安装：`pip install -r requirements.txt`
- 检查端口8000是否被占用


### 2. 文档上传失败怎么办？

**问题**：上传文件时提示错误
**解决**：
- 检查文件格式是否支持
- 检查文件名是否包含特殊字符

### 3. 检索结果为空怎么办？

**问题**：检索时没有任何结果返回
**解决**：
- 确认文档已成功上传到资产库
- 检查检索关键词是否准确
- 检查检索字段类型

### 4. 服务挂掉怎么办？

如果服务因为各种原因挂掉了，只需重新启动，系统自动加载所有底层库表数据和bank，同时绑定所有接口
```
python app/main.py
```
