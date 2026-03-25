from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class PageRef:
    page_id: int
    slot_count: int


@dataclass
class RequestPageTable:
    request_id: str
    pages: List[PageRef] = field(default_factory=list)
    total_tokens: int = 0


class PagedKVCacheManager:
    def __init__(self, page_size_tokens: int = 16, max_pages: int = 4096):
        self.page_size_tokens = page_size_tokens
        self.max_pages = max_pages
        self.free_pages = list(range(max_pages))
        self.page_tables: Dict[str, RequestPageTable] = {}

    def allocate_for_tokens(self, request_id: str, num_new_tokens: int) -> None:
        table = self.page_tables.setdefault(request_id, RequestPageTable(request_id=request_id))
        remaining = num_new_tokens
        while remaining > 0:
            if not self.free_pages:
                raise RuntimeError("Out of KV pages")
            page_id = self.free_pages.pop()
            slot_count = min(remaining, self.page_size_tokens)
            table.pages.append(PageRef(page_id=page_id, slot_count=slot_count))
            table.total_tokens += slot_count
            remaining -= slot_count

    def release_request(self, request_id: str) -> None:
        table = self.page_tables.get(request_id)
        if table is None:
            return
        for page in table.pages:
            self.free_pages.append(page.page_id)
        del self.page_tables[request_id]

    def stats(self):
        used = self.max_pages - len(self.free_pages)
        return {
            "page_size_tokens": self.page_size_tokens,
            "max_pages": self.max_pages,
            "used_pages": used,
            "free_pages": len(self.free_pages),
            "active_requests": len(self.page_tables),
        }

    def get_request_pages(self, request_id: str) -> Optional[RequestPageTable]:
        return self.page_tables.get(request_id)