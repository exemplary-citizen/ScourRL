from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class Evidence(BaseModel):
    url: HttpUrl
    quote: str = Field(min_length=2)
    supports: str = Field(min_length=2)


class PurchasePacket(BaseModel):
    query: str
    recommended_product: str
    retailer: str
    url: HttpUrl
    price: str | None = None
    delivery_or_pickup: str | None = None
    seller: str | None = None
    constraints_met: list[str] = Field(default_factory=list)
    constraints_uncertain: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    recommendation: str
    stop_before_checkout: bool = True
    cart_prepared: bool = False
    attempted_checkout: bool = False


class ShoppingTaskSpec(BaseModel):
    task_id: str
    instruction: str
    allowed_domains: list[str]
    max_price: float
    must_have: list[str] = Field(default_factory=list)
    must_not_have: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(
        default_factory=lambda: ["checkout", "payment", "login"]
    )
    token_budget: int = 750
    require_cart: bool = False


class PageSnapshot(BaseModel):
    url: HttpUrl
    title: str | None = None
    price_text: str | None = None
    page_text_snippets: list[str] = Field(default_factory=list)
    timestamp: str | None = None
