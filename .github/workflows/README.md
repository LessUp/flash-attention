# GitHub Workflow 打标流程

本仓库使用独立的 tag 通道，避免 FA2 和 FA4 的发布相互冲突。

## 发布通道

| Tag 模式 | Workflow | 包目标 | 版本来源 |
| --- | --- | --- | --- |
| `v*` | `.github/workflows/publish.yml` | 根包（`flash-attn`） | 根包版本元数据 |
| `fa4-v*` | `.github/workflows/publish-fa4.yml` | `flash_attn/cute` 包（`flash-attn-4`） | 使用 `fa4-v*` tags 的 `setuptools-scm` |

## 如何发布

### FA2 / 根包通道

1. 创建一个匹配 `v*` 的 tag（示例：`v2.9.0`）。
2. 推送该 tag。
3. `publish.yml` 会创建 release、构建 wheel matrix 产物，并发布到 PyPI。

### FA4 / CUTE 包通道

**手动发布**：创建并推送一个匹配 `fa4-v*` 的 tag（示例：`fa4-v4.0.0`）。

**每周 beta**：`publish-fa4.yml` 还会通过 cron 在每周三 08:00 UTC 运行。定时或手动运行会创建并推送下一个 `fa4-v*.beta*` tag，然后在同一个 workflow 运行中继续构建并发布这个 beta。手动触发被限制在仓库默认分支上，因此不会给特性分支的提交打 tag。推送的 tag 匹配 `fa4-v*` 触发条件，但 GitHub 会抑制由 `GITHUB_TOKEN` 创建的事件触发的 workflow 运行，所以不会产生递归运行。

| 周 | 创建的 tag | PyPI 版本 |
| --- | --- | --- |
| 1 | `fa4-v4.0.0.beta5` | `4.0.0b5` |
| 2 | `fa4-v4.0.0.beta6` | `4.0.0b6` |

要停止每周 beta：GitHub 仓库 → Actions → "Publish flash-attn-4 to PyPI" → `···` 菜单 → **Disable workflow**。需要恢复时重新启用，或者移除 `schedule` 触发条件、只用手动打 tag。用户仍然可以在计划之外需要发布 beta 时直接推送 `fa4-v*.beta*` tag。

## 护栏（Guardrails）

- 不要为 FA4 发布使用 `v*` tags。
- 不要为 FA2 发布使用 `fa4-v*` tags。
- 保持 `flash_attn/cute/pyproject.toml` 的 tag 解析与 FA4 tag 前缀同步。
- workflow 文件名（`publish-fa4.yml`）是 PyPI trusted publishing OIDC 身份的一部分——重命名时必须同步更新 PyPI。
