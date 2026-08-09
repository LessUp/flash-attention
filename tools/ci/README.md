# FA4 CI

CI 运行在自托管的 GPU runner 上，使用从 Docker Hub 拉取的 Apptainer（SIF）容器。
每次推送到 `main` 时触发。

## 两遍测试策略

- **第一遍** — 通过 `FakeTensorMode` 并行编译 kernel（不需要 GPU 内存）
- **第二遍** — 在真实 GPU 上使用缓存的已编译 kernel 运行测试

参见 `run_fa4_ci.py`，它包含了 CI 和 `test_ci_local.sh` 共用的逻辑。

## 需要的 GitHub secrets / variables

| 名称 | 种类 | 值 |
|------|------|-------|
| `DOCKERHUB_USERNAME` | Secret | Docker Hub 用户名 |
| `DOCKERHUB_TOKEN` | Secret | Docker Hub 访问令牌 |
| `CI_WORK_DIR` | Variable | runner 上的大磁盘路径，例如 `/scratch/user/johnson` |

`CI_WORK_DIR` 用于 SIF 缓存和 Apptainer 临时文件。如果未设置，回退到 `/scratch/user/<github-actor>`。

## 更新容器镜像

1. 通过 `tools/ci/docker/build.sh` + `tag_and_push.sh` 构建并推送新镜像。
2. 用新 tag 和 `sha256` 摘要更新 `.github/workflows/ci.yml` 中的 `FA4_IMAGE`。
3. 旧的 SIF 会在下一次 CI 运行时自动从 runner 上删除。

## 扩展测试覆盖

编辑 `.github/workflows/ci.yml` 中的 `FA4_TEST_FILTER`。要运行完整测试套件，把它设为空字符串，并增加 `gpu-test` action 调用中的 `compile-workers`。

另外，可以编辑 `run_fa4_ci.py` 来修改 `DEFAULT_TEST_TARGET` 或 worker 默认值——那里的修改对 CI 和本地运行都生效。

## FA2 导入隔离

测试在 Apptainer 容器内运行。仓库的 `flash_attn/__init__.py` 会导入 FA2 的 C 扩展（`flash_attn_2_cuda`），而容器中不存在这个扩展。`run_fa4_ci.py` 通过以下方式解决：

1. 运行时把当前仓库中的 FA4 安装进容器（`uv pip install -e flash_attn/cute`）。
2. 从 `/tmp` 用绝对测试路径运行 pytest——这样仓库根目录不会出现在 `sys.path[0]` 中，因此找到的是已安装的 FA4 包，而不是 FA2 的 `__init__.py`。

`flash_attn/__init__.py` 有意不做修改；隔离完全在 CI 中处理。

## 添加新的 runner / GPU 类型

1. 在机器上注册一个自托管 runner，并设置想要的标签（例如 `h100`）。
2. 把该标签添加到 `.github/workflows/ci.yml` 的 `gpu` matrix 中。
3. 如果新机器的 scratch 路径不同，为它设置 `CI_WORK_DIR`。
