"""Cross-encoder reranking: local BAAI bge-reranker or DashScope qwen3 API.

Hybrid recall (dense+BM25+RRF) buys recall; a cross-encoder re-scores the
top pool with full query-document attention, which is the single cheapest
large win for ranking quality.  Two providers share one interface:

- ``local``: the classic CPU cross-encoder (BAAI/bge-reranker-base).  The
  model loads lazily and offline-first (same dance as the embedding model);
  when the local cache is cold the download falls back to the hf-mirror
  endpoint so it works without a proxy.
- ``dashscope``: Bailian's hosted rerank HTTP API (qwen3-vl-rerank and the
  gte/qwen3 rerank family).  Scores come back as 0.0-1.0 relevance, so no
  model download is needed.  Any API failure - missing key, network error,
  timeout, malformed payload - degrades to the local cross-encoder instead
  of failing the query, so ranking never hard-fails on the network.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_DASHSCOPE_MODEL = "qwen3-vl-rerank"
DEFAULT_DASHSCOPE_BASE_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


class RerankerModule:
    """Rerank a candidate pool with a cross-encoder; adds ``rerank_score``."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        enabled: bool = True,
        pool_size: int = 20,
        device: str = "cpu",
        model: Any | None = None,
        provider: str = "local",
        api_key: str = "",
        base_url: str = DEFAULT_DASHSCOPE_BASE_URL,
        timeout_seconds: float = 10.0,
        include_context: bool = False,
    ):
        if pool_size < 1:
            raise ValueError("pool_size 必须为正整数")
        provider = provider.strip().lower()
        if provider not in {"local", "dashscope"}:
            raise ValueError(f"未知的 rerank provider: {provider!r}（可选 local / dashscope）")
        self.provider = provider
        self.model_name = model_name
        self.enabled = enabled
        self.pool_size = pool_size
        self.device = device
        self._model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.include_context = include_context

    # ------------------------------------------------------------------ load

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        import huggingface_hub.constants as hf_constants

        forced_offline = os.getenv("RAG_HF_OFFLINE", "").casefold() in {"1", "true", "yes", "on"}
        previous_offline = hf_constants.HF_HUB_OFFLINE
        hf_constants.HF_HUB_OFFLINE = True
        os.environ["HF_HUB_OFFLINE"] = "1"

        from sentence_transformers import CrossEncoder

        try:
            self._model = CrossEncoder(self.model_name, device=self.device)
        except Exception as offline_error:
            if forced_offline:
                raise
            hf_constants.HF_HUB_OFFLINE = previous_offline
            os.environ.pop("HF_HUB_OFFLINE", None)
            logger.warning(
                "离线加载重排模型失败（%s），改为在线下载（约 1.1GB，仅首次）",
                str(offline_error)[:200],
            )
            try:
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as online_error:
                # 国内直连 HF 常被墙：切 hf-mirror 再试一次。ENDPOINT 与
                # URL 模板都在 huggingface_hub 导入时固化，需一并重写。
                if os.getenv("HF_ENDPOINT"):
                    raise RuntimeError(
                        f"重排模型下载失败（HF_ENDPOINT={os.getenv('HF_ENDPOINT')}）。"
                        "请检查网络/代理后重试，或设置 RAG_RERANK_ENABLED=false 跳过重排"
                    ) from online_error
                os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
                hf_constants.ENDPOINT = HF_MIRROR_ENDPOINT
                hf_constants.HUGGINGFACE_CO_URL_TEMPLATE = (
                    HF_MIRROR_ENDPOINT + "/{repo_id}/resolve/{revision}/{filename}"
                )
                logger.info("切换 HF 镜像端点重试下载: %s", HF_MIRROR_ENDPOINT)
                try:
                    self._model = CrossEncoder(self.model_name, device=self.device)
                except Exception as mirror_error:  # noqa: BLE001 - 给出可操作的指引
                    raise RuntimeError(
                        "重排模型下载失败：直连与 hf-mirror 均不可用。"
                        "解决办法（任选其一）：1) 开启代理后重跑；"
                        "2) 在 .env 设置 HF_ENDPOINT=https://hf-mirror.com 后重跑；"
                        "3) 设置 RAG_RERANK_ENABLED=false 暂时关闭重排"
                    ) from mirror_error
        logger.info("重排模型已加载: %s（设备: %s）", self.model_name, self.device)
        return self._model

    # ---------------------------------------------------------------- rerank

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Return documents sorted by relevance, annotated with rerank_score."""
        if not self.enabled or not documents:
            return documents
        texts = [self._scoring_text(doc) for doc in documents]
        if self.provider == "dashscope":
            scores = self._dashscope_scores(query, texts)
            if scores is not None:
                logger.debug("DashScope 重排成功: model=%s, n=%d", self.model_name, len(documents))
                return self._annotate_and_sort(documents, scores)
            logger.warning("DashScope 重排不可用，回退本地 cross-encoder 继续排序")
        model = self._load_model()
        pairs = [(query, text) for text in texts]
        scores = [float(score) for score in model.predict(pairs)]
        return self._annotate_and_sort(documents, scores)

    def _scoring_text(self, document: Document) -> str:
        """Build the text the reranker actually sees.

        Markdown 分块把菜名标题留在首个 child 里，后续片段（步骤/小贴士）
        自描述不完整；include_context 开启时把 parent 级菜名（分类）拼进
        打分文本，等价于企业方案一的 title 注入。
        """
        if not self.include_context:
            return document.page_content
        dish = str(document.metadata.get("dish_name") or "").strip()
        if not dish:
            return document.page_content
        category = str(document.metadata.get("category") or "").strip()
        title = f"{dish}（{category}）" if category and category not in {"其他", "未知"} else dish
        return f"{title}｜{document.page_content}"

    @staticmethod
    def _annotate_and_sort(documents: list[Document], scores: list[float]) -> list[Document]:
        for document, score in zip(documents, scores, strict=True):
            document.metadata["rerank_score"] = score
        return sorted(documents, key=lambda doc: doc.metadata["rerank_score"], reverse=True)

    # ------------------------------------------------------------- dashscope

    def _dashscope_scores(self, query: str, texts: list[str]) -> list[float] | None:
        """Call the Bailian rerank API; per-input scores, or None on any failure."""
        if not self.api_key:
            logger.warning("未设置 DASHSCOPE_API_KEY，无法使用云端重排")
            return None
        payload = {
            "model": self.model_name,
            "input": {"query": query, "documents": texts},
            "parameters": {"top_n": len(texts), "return_documents": False},
        }
        last_error = ""
        for attempt in (1, 2):
            try:
                import requests

                response = requests.post(
                    self.base_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.timeout_seconds,
                )
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                else:
                    scores = self._parse_dashscope_results(response.json(), len(texts))
                    if scores is not None:
                        return scores
                    last_error = f"响应结构与预期不符: {str(response.text)[:200]}"
            except Exception as error:  # noqa: BLE001 - 任何失败都走本地回退
                last_error = str(error)[:200]
            logger.warning("DashScope 重排第 %d 次调用失败: %s", attempt, last_error)
        logger.error("DashScope 重排重试后仍失败，回退本地模型: %s", last_error)
        return None

    @staticmethod
    def _parse_dashscope_results(data: Any, expected: int) -> list[float] | None:
        """Extract ``relevance_score`` per input index; None unless a full permutation."""
        if not isinstance(data, dict):
            return None
        body = data.get("output") if isinstance(data.get("output"), dict) else data
        results = body.get("results")
        if not isinstance(results, list) or len(results) != expected:
            return None
        scores: list[float | None] = [None] * expected
        for item in results:
            try:
                index = int(item["index"])
                scores[index] = float(item["relevance_score"])
            except (KeyError, TypeError, ValueError, IndexError):
                return None
        if any(score is None for score in scores):
            return None
        return [float(score) for score in scores]
