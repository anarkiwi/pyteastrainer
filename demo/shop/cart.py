"""Tiny shopping-cart module used by the demo suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


class CartError(Exception):
    pass


@dataclass
class Item:
    sku: str
    price: float
    qty: int = 1


@dataclass
class Cart:
    items: Dict[str, Item] = field(default_factory=dict)
    coupon: str | None = None

    def add(self, sku: str, price: float, qty: int = 1) -> None:
        if qty <= 0:
            raise CartError("qty must be positive")
        if price < 0:
            raise CartError("price must be non-negative")
        if sku in self.items:
            self.items[sku].qty += qty
        else:
            self.items[sku] = Item(sku, price, qty)

    def remove(self, sku: str) -> None:
        if sku not in self.items:
            raise CartError(f"no such sku: {sku}")
        del self.items[sku]

    def subtotal(self) -> float:
        return sum(i.price * i.qty for i in self.items.values())

    def apply_coupon(self, code: str) -> None:
        if code not in COUPONS:
            raise CartError("unknown coupon")
        self.coupon = code

    def total(self) -> float:
        sub = self.subtotal()
        if self.coupon:
            kind, value = COUPONS[self.coupon]
            if kind == "pct":
                sub *= 1 - value / 100
            else:
                sub = max(0.0, sub - value)
        if sub >= 100:
            return round(sub, 2)  # free shipping
        return round(sub + 5.0, 2)

    def skus(self) -> List[str]:
        return sorted(self.items)


COUPONS = {"TEN": ("pct", 10), "FIVEOFF": ("abs", 5)}
