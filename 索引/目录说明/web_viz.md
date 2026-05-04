# 目录说明：`web_viz`

- `index_v2.html` 存在：`True`
- `index_v2.html` 直接引用 BFF：`True`

## 当前模板
| template | role |
| --- | --- |
| index.html | legacy/full page |
| index_v2.html | BFF template |
| paper_demo.html | paper demo |

## 当前口径
- `paper_demo.html` 对应论文/单 case 演示页。
- `index_v2.html` 已接 `/api/v1/visualization/case_data`，但是否作为默认首页仍应单独判断。
