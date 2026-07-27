# Prompt 契约（Week 2）

## system / user / tool 分工

| 角色 | 职责 |
|------|------|
| system | 身份、硬规则、输出合同（JSON 字段、禁止废话） |
| user | 具体任务；校验失败时的 **repair** 指令 |
| tool | （本 demo 未用）参数 schema = 工具侧契约；与 JSON 最终答案契约是同一思想 |

## 不可靠组件模型

1. 调用模型  
2. **校验**（parse + schema）  
3. 失败 → **至多一次** repair user 消息  
4. 仍失败 → 对调用方抛错 / 返回 halted（不要假装成功）

## 和 research_agent 迁移的关系

week03 练的是「最终答案 JSON 合同」。  
research_agent 迁的是「最终答案必须引用真实相对路径」——校验换了，**重试回路相同**。
