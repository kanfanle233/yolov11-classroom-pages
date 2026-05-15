# 目录说明：`server`

- 角色：FastAPI 服务入口，负责 bundle API、paper demo 路由和 BFF 聚合接口。

## 当前接口
| interface | status |
| --- | --- |
| /api/bundle/list | ok |
| /api/bundle/{case_id}/manifest | ok |
| /api/bundle/{case_id}/verified | ok |
| /api/v1/visualization/case_data | ok |
| /paper/bundle/{case_id} | ok |

## 当前口径
- `server/app.py` 已有 `/api/bundle/*`。
- `server/app.py` 已有 `/api/v1/visualization/case_data`。
- `paper_demo.html` 是现有单 case 展示页，`index_v2.html` 是已接 BFF 的新模板。
