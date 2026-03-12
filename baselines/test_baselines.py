"""Sanity check: verify all baselines import correctly and standalone components work."""
import sys
import os
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

passed = 0
total = 0


def assert_(cond):
    if not cond:
        raise AssertionError("Assertion failed")


def test(name, fn):
    global passed, total
    total += 1
    try:
        fn()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")


# ── Import tests ──────────────────────────────────────────────────────
print("=== Import Tests ===")
test("import baselines", lambda: __import__("baselines"))
test("import sparse_rl", lambda: __import__("baselines.sparse_rl"))
test("import qurl", lambda: __import__("baselines.qurl"))
test("import rlhfless", lambda: __import__("baselines.rlhfless"))
test("import r_kv", lambda: __import__("baselines.r_kv"))

# ── Registry test ─────────────────────────────────────────────────────
print("\n=== Registry ===")
from baselines import BASELINE_REGISTRY
test("registry has 4 methods", lambda: assert_(len(BASELINE_REGISTRY) == 4))
test("all callables", lambda: assert_(all(callable(v) for v in BASELINE_REGISTRY.values())))

# ── QuRL UAQ scaling ──────────────────────────────────────────────────
print("\n=== QuRL UAQ Scaling ===")
from baselines.qurl import apply_uaq_scaling, revert_uaq_scaling

def test_uaq():
    model = torch.nn.Sequential(
        torch.nn.LayerNorm(64),
        torch.nn.Linear(64, 64, bias=True),
    )
    w_orig = model[1].weight.data.clone()
    ln_orig = model[0].weight.data.clone()
    apply_uaq_scaling(model, scale=1.5)
    assert torch.allclose(model[1].weight.data, w_orig / 1.5, atol=1e-5)
    assert torch.allclose(model[0].weight.data, ln_orig * 1.5, atol=1e-5)
    revert_uaq_scaling(model, scale=1.5)
    assert torch.allclose(model[1].weight.data, w_orig, atol=1e-4)
    assert torch.allclose(model[0].weight.data, ln_orig, atol=1e-4)

test("UAQ scale + revert roundtrip", test_uaq)

# ── RLHFless scheduling components ───────────────────────────────────
print("\n=== RLHFless Scheduling ===")
from baselines.rlhfless import (
    LengthPredictor, find_shared_prefix, assign_prompts_by_length,
    should_cut_and_migrate,
)

def test_length_predictor():
    lp = LengthPredictor()
    pred = lp.predict(42)
    assert pred == 128  # default
    lp.update(42, 200)
    assert lp.predict(42) == 200  # EWMA alpha=1

test("LengthPredictor", test_length_predictor)

def test_shared_prefix():
    prompts = [[1, 2, 3, 4, 5], [1, 2, 3, 6, 7], [1, 2, 3, 8, 9]]
    assert find_shared_prefix(prompts) == 3

test("find_shared_prefix", test_shared_prefix)

def test_assignment():
    assignments = assign_prompts_by_length(["a", "b", "c", "d"], [100, 50, 200, 75], 2)
    assert len(assignments) == 2
    assert sum(len(a) for a in assignments) == 4

test("assign_prompts_by_length", test_assignment)

def test_cut_migrate():
    assert not should_cut_and_migrate([True, False, False, False])
    assert should_cut_and_migrate([True, True, True, False])

test("should_cut_and_migrate", test_cut_migrate)

# ── R-KV compression engine ──────────────────────────────────────────
print("\n=== R-KV Compression Engine ===")
from baselines.r_kv import RKVCacheCompressor

def test_rkv_compressor():
    comp = RKVCacheCompressor(budget=32, buffer_size=16, alpha=4, beta=2, lam=0.1)
    num_keys = 64
    head_dim = 32
    num_heads = 4

    keys = torch.randn(num_keys, head_dim)
    attn = torch.softmax(torch.randn(num_heads, 8, num_keys), dim=-1)

    keep_idx = comp.compute_eviction(keys, attn)
    assert keep_idx.shape[0] <= comp.budget
    assert keep_idx.shape[0] >= comp.alpha  # At least protected tokens
    # Protected tokens should be in keep set
    for idx in range(num_keys - comp.alpha, num_keys):
        assert idx in keep_idx

test("RKVCacheCompressor eviction", test_rkv_compressor)

def test_rkv_no_eviction():
    comp = RKVCacheCompressor(budget=100)
    keys = torch.randn(50, 32)
    attn = torch.softmax(torch.randn(4, 8, 50), dim=-1)
    keep_idx = comp.compute_eviction(keys, attn)
    assert keep_idx.shape[0] == 50  # No eviction needed

test("RKVCacheCompressor no eviction", test_rkv_no_eviction)

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {passed}/{total} tests passed")
if passed == total:
    print("🎉 ALL TESTS PASSED!")
else:
    print("⚠️  Some tests failed!")
    sys.exit(1)


def assert_(cond):
    if not cond:
        raise AssertionError("Assertion failed")
