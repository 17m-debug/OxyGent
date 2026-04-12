## 评价创建与管理

### 1. 创建评价（含评论）

创建或更新对话评价，支持附加评论。

**接口信息**
- **路径**: `POST /rating`
- **说明**: 为指定对话创建评价记录，可包含点赞/点踩类型和可选的文字评论

**请求参数**

```json
{
  "trace_id": "string",        // 必填，对话追踪ID
  "rating_type": "like|dislike", // 必填，评价类型：like(点赞) 或 dislike(点踩)
  "comment": "string",         // 可选，评论内容
  "erp": "string"              // 可选，ERP系统标识
}
```

**请求示例**

```bash
curl -X POST "http://localhost:8000/rating" \
  -H "Content-Type: application/json" \
  -d '{
    "trace_id": "abc123",
    "rating_type": "like",
    "comment": "回答很准确，解决了我的问题！"
  }'
```

**响应示例**

```json
{
  "code": 200,
  "message": "Rating successful",
  "data": {
    "rating_id": "uuid-xxx-xxx",
    "stats": {
      "trace_id": "abc123",
      "like_count": 5,
      "dislike_count": 1,
      "total_ratings": 6,
      "satisfaction_rate": 83.33,
      "last_updated": "2025-12-29 10:30:45.123456"
    }
  }
}
```

**字段说明**
- `rating_id`: 新创建的评价记录ID
- `stats`: 更新后的统计数据
  - `like_count`: 点赞总数
  - `dislike_count`: 点踩总数
  - `total_ratings`: 总评价数
  - `satisfaction_rate`: 满意度（点赞比例）
  - `last_updated`: 最后更新时间

---

### 2. 删除评价

删除指定的评价记录（管理员功能）。

**接口信息**
- **路径**: `DELETE /rating/{rating_id}`
- **说明**: 通过 rating_id 删除评价记录并重新计算统计数据

**路径参数**
- `rating_id`: 评价记录ID

**请求示例**

```bash
curl -X DELETE "http://localhost:8000/rating/uuid-xxx-xxx"
```

**响应示例**

```json
{
  "code": 200,
  "message": "Rating deleted successfully",
  "data": {
    "deleted": true,
    "rating_id": "uuid-xxx-xxx"
  }
}
```

---

## 评价查询

### 3. 获取评价统计

获取指定对话的评价统计信息。

**接口信息**
- **路径**: `GET /rating/{trace_id}`
- **说明**: 查询对话的评价汇总统计

**路径参数**
- `trace_id`: 对话追踪ID

**请求示例**

```bash
curl "http://localhost:8000/rating/abc123"
```

**响应示例**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "trace_id": "abc123",
    "like_count": 5,
    "dislike_count": 1,
    "total_ratings": 6,
    "satisfaction_rate": 83.33,
    "last_updated": "2025-12-29 10:30:45.123456"
  }
}
```

---

### 4. 获取当前评价

获取对话的最新一条评价记录。

**接口信息**
- **路径**: `GET /rating/{trace_id}/current`
- **说明**: 返回指定对话的最新评价记录（按创建时间）

**路径参数**
- `trace_id`: 对话追踪ID

**查询参数**
- `erp`: (可选) ERP系统标识，用于过滤特定ERP的评价

**请求示例**

```bash
curl "http://localhost:8000/rating/abc123/current?erp=system1"
```

**响应示例**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "trace_id": "abc123",
    "current_rating": {
      "rating_id": "uuid-xxx-xxx",
      "trace_id": "abc123",
      "rating_type": "like",
      "user_id": null,
      "user_ip": "192.168.1.100",
      "comment": "回答很准确，解决了我的问题！",
      "erp": "system1",
      "create_time": "2025-12-29 10:30:45.123456",
      "update_time": null
    }
  }
}
```
---

## 错误码说明

| 错误码 | 说明 |
|-------|------|
| 200 | 成功 |
| 400 | 请求参数错误或操作失败 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

---

## 注意事项

1. **评论长度限制**：建议评论内容不超过 1000 字符
2. **多次评价**：同一对话可以多次评价，系统会保留所有历史记录
3. **评论可选性**：`comment` 字段是可选的，可以仅评价不评论
4. **统计实时性**：评价统计会在创建/删除评价后立即更新
5. **索引刷新**：系统会自动刷新索引以确保数据可立即查询
6. **ERP 集成**：使用 `erp` 字段可以区分不同用户的评价