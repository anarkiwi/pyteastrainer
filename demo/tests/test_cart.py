"""A suite in the style an over-eager coding agent produces.

Legit tests are at the top. Everything under "BLOAT" is the kind of thing
that accumulates: paraphrased copies, same-assertion-different-literal
clones, and "smaller" tests that a bigger one already proves.
"""
import pytest

from shop.cart import Cart, CartError


# ----------------------------------------------------------------- legit --
def test_add_new_item():
    c = Cart()
    c.add("a", 10.0)
    assert c.skus() == ["a"]
    assert c.items["a"].qty == 1


def test_add_existing_item_increments_qty():
    c = Cart()
    c.add("a", 10.0)
    c.add("a", 10.0, qty=2)
    assert c.items["a"].qty == 3


def test_add_rejects_zero_qty():
    with pytest.raises(CartError):
        Cart().add("a", 1.0, qty=0)


def test_add_rejects_negative_price():
    with pytest.raises(CartError):
        Cart().add("a", -1.0)


def test_remove_unknown_sku():
    with pytest.raises(CartError):
        Cart().remove("zzz")


def test_remove():
    c = Cart()
    c.add("a", 1.0)
    c.remove("a")
    assert c.skus() == []


def test_total_adds_shipping_under_100():
    c = Cart()
    c.add("a", 40.0)
    assert c.total() == 45.0


def test_total_free_shipping_at_100():
    c = Cart()
    c.add("a", 100.0)
    assert c.total() == 100.0


def test_pct_coupon():
    c = Cart()
    c.add("a", 200.0)
    c.apply_coupon("TEN")
    assert c.total() == 180.0


def test_abs_coupon_floors_at_zero():
    c = Cart()
    c.add("a", 2.0)
    c.apply_coupon("FIVEOFF")
    assert c.total() == 5.0


def test_unknown_coupon():
    with pytest.raises(CartError):
        Cart().apply_coupon("NOPE")


@pytest.mark.strain_exempt(reason="checks float rounding, not a path")
def test_total_rounds_to_cents():
    c = Cart()
    c.add("a", 33.333, qty=3)
    assert c.total() == 99.999 + 5.0 or c.total() == 105.0


# ----------------------------------------------------------------- BLOAT --
# 1. Paraphrased copies of test_add_new_item (same body, renamed variables).
def test_add_single_item():
    cart = Cart()
    cart.add("a", 10.0)
    assert cart.skus() == ["a"]
    assert cart.items["a"].qty == 1


def test_adding_item_to_empty_cart():
    """Verify that adding an item to an empty cart works."""
    the_cart = Cart()
    the_cart.add("a", 10.0)
    assert the_cart.skus() == ["a"]
    assert the_cart.items["a"].qty == 1


# 2. Clones: same path and same assertion shape, different literals.
def test_total_with_shipping_small():
    c = Cart()
    c.add("x", 10.0)
    assert c.total() == 15.0


def test_total_with_shipping_medium():
    c = Cart()
    c.add("y", 60.0)
    assert c.total() == 65.0


def test_total_with_shipping_tiny():
    c = Cart()
    c.add("z", 0.5)
    assert c.total() == 5.5


# 3. Subsumed: everything these touch, test_pct_coupon already touches.
def test_coupon_is_stored():
    c = Cart()
    c.apply_coupon("TEN")
    assert c.coupon == "TEN"


def test_subtotal_single():
    c = Cart()
    c.add("a", 3.0)
    assert c.subtotal() == 3.0


# 4. Parametrize with duplicate rows (identical values) and clone rows.
@pytest.mark.parametrize("price,expected", [
    (10.0, 15.0),
    (20.0, 25.0),
    (10.0, 15.0),      # agent re-added an existing case
])
def test_total_param(price, expected):
    c = Cart()
    c.add("a", price)
    assert c.total() == expected
