#!/usr/bin/env python3
"""Test CKKS bootstrapping in PhantomFHE."""

import sys
sys.path.insert(0, 'build/lib')
import pyPhantom as ph
import numpy as np
import time


def test_bootstrap(poly_degree=32768, level_budget=None, special_mod_size=3,
                   L0=23, bit_size=59):
    if level_budget is None:
        level_budget = [2, 2]
    slots_target = poly_degree // 2
    galois_elts = ph.ckks_bootstrapper.get_galois_elements(
        poly_degree, 0, level_budget)

    parms = ph.params(ph.scheme_type.ckks)
    parms.set_poly_modulus_degree(poly_degree)
    parms.set_special_modulus_size(special_mod_size)
    parms.set_galois_elts(galois_elts)

    num_primes = L0 + special_mod_size
    coeff_modulus = ph.create_coeff_modulus(poly_degree, [bit_size] * num_primes)
    parms.set_coeff_modulus(coeff_modulus)

    context = ph.context(parms)
    secret_key = ph.secret_key(context)
    encoder = ph.ckks_encoder(context)
    slots = encoder.slot_count()

    bt = ph.ckks_bootstrapper(encoder)
    bt.setup(context, level_budget)
    bt.keygen(context, secret_key)

    scale = 2 ** bit_size
    np.random.seed(42)
    test_vals = [complex(v, 0.0) for v in np.random.uniform(1.0, 5.0, slots)]

    pt = encoder.encode_complex_vector(context, test_vals, scale)
    ct = secret_key.encrypt_symmetric(context, pt)

    levels_to_consume = L0 - 3
    for _ in range(levels_to_consume):
        ct = ph.mod_switch_to_next(context, ct)

    t0 = time.time()
    ct_refreshed = bt.bootstrap(context, ct)
    bt_time = time.time() - t0

    pt_after = secret_key.decrypt(context, ct_refreshed)
    vals_after = encoder.decode_complex_vector(context, pt_after)

    errors = [abs(vals_after[i].real - test_vals[i].real) for i in range(slots)]
    max_err = max(errors)
    avg_err = sum(errors) / len(errors)

    print(f"N={poly_degree}, P={special_mod_size}, L0={L0}, bits={bit_size}")
    print(f"  Bootstrap time: {bt_time:.1f}s")
    print(f"  Max error: {max_err:.6e}")
    print(f"  Avg error: {avg_err:.6e}")

    for i in range(5):
        print(f"    [{i}] orig={test_vals[i].real:.7f}  got={vals_after[i].real:.7f}  "
              f"err={errors[i]:.6e}")

    assert max_err < 0.1, f"Bootstrap error too large: {max_err}"
    print("  PASS")
    return max_err


if __name__ == "__main__":
    test_bootstrap()
