#pragma once

#include <cmath>
#include <complex>
#include <vector>
#include <map>
#include <memory>

#include "ciphertext.h"
#include "context.cuh"
#include "ckks.h"
#include "ntt.cuh"
#include "plaintext.h"
#include "secretkey.h"
#include "evaluate.cuh"
#include "cuda_wrapper.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace phantom {

namespace boot_params {
enum {
    LEVEL_BUDGET,
    LAYERS_COLL,
    LAYERS_REM,
    NUM_ROTATIONS,
    BABY_STEP,
    GIANT_STEP,
    NUM_ROTATIONS_REM,
    BABY_STEP_REM,
    GIANT_STEP_REM,
    TOTAL_ELEMENTS
};
}

struct BootPrecom {
    uint32_t dim1 = 0;
    uint32_t slots = 0;
    std::vector<int32_t> params_enc = std::vector<int32_t>(boot_params::TOTAL_ELEMENTS, 0);
    std::vector<int32_t> params_dec = std::vector<int32_t>(boot_params::TOTAL_ELEMENTS, 0);
    std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>> enc_precom;
    std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>> dec_precom;
};

class CKKSBootstrapper {
public:
    CKKSBootstrapper(PhantomCKKSEncoder& encoder) : encoder_(encoder) {}
    ~CKKSBootstrapper() = default;

    void setup(const PhantomContext& context,
               std::vector<uint32_t> level_budget,
               std::vector<uint32_t> dim1 = {0, 0},
               uint32_t num_slots = 0,
               uint32_t correction_factor = 0);

    void keygen(const PhantomContext& context,
                PhantomSecretKey& secret_key,
                uint32_t num_slots = 0);

    PhantomCiphertext bootstrap(const PhantomContext& context,
                                PhantomCiphertext& ct,
                                uint32_t num_slots = 0);

    // Debug version with intermediate decryption
    PhantomCiphertext bootstrap_debug(const PhantomContext& context,
                                      PhantomCiphertext& ct,
                                      PhantomSecretKey& sk,
                                      uint32_t num_slots = 0);

    static uint32_t get_bootstrap_depth(const std::vector<uint32_t>& level_budget);

    static std::vector<uint32_t> get_galois_elements(
        uint32_t poly_degree, uint32_t num_slots,
        const std::vector<uint32_t>& level_budget);

    const PhantomRelinKey& get_relin_key() const { return relin_key_; }
    const PhantomGaloisKey& get_galois_key() const { return galois_key_; }

    PhantomCiphertext coeffs_to_slots(const PhantomContext& context,
                                      const PhantomCiphertext& ct,
                                      uint32_t num_slots = 0);

    PhantomCiphertext slots_to_coeffs(const PhantomContext& context,
                                      const PhantomCiphertext& ct,
                                      uint32_t num_slots = 0);

private:
    PhantomCiphertext mod_raise(const PhantomContext& context, PhantomCiphertext& ct);

    PhantomCiphertext eval_chebyshev(const PhantomContext& context,
                                     const PhantomCiphertext& ct,
                                     const std::vector<double>& coeffs,
                                     double a, double b);

    void double_angle_iterations(const PhantomContext& context,
                                 PhantomCiphertext& ct,
                                 uint32_t num_iter);

    void adjust_ciphertext(const PhantomContext& context,
                           PhantomCiphertext& ct,
                           double correction);

    void mult_by_const(const PhantomContext& context,
                       PhantomCiphertext& ct,
                       double scalar);

    void mult_by_integer(const PhantomContext& context,
                         PhantomCiphertext& ct,
                         uint64_t integer);

    void mult_by_monomial(const PhantomContext& context,
                          PhantomCiphertext& ct,
                          uint32_t power);

    void add_const(const PhantomContext& context,
                   PhantomCiphertext& ct,
                   double scalar);

    PhantomCiphertext conjugate(const PhantomContext& context,
                                const PhantomCiphertext& ct);

    std::vector<int32_t> find_rotation_indices(uint32_t slots, uint32_t M);

    // BSGS linear transform using standard rotate/multiply_plain API
    PhantomCiphertext bsgs_linear_transform(
        const PhantomContext& context,
        const PhantomCiphertext& ct,
        const std::vector<std::shared_ptr<PhantomPlaintext>>& diag_precom,
        int32_t num_rotations, int32_t b, int32_t g,
        const std::vector<int32_t>& rot_in,
        const std::vector<int32_t>& rot_out);

    // Hoisted BSGS linear transform
    PhantomCiphertext bsgs_linear_transform_hoisted(
        const PhantomContext& context,
        const PhantomCiphertext& ct,
        const std::vector<std::shared_ptr<PhantomPlaintext>>& diag_precom,
        int32_t num_rotations, int32_t b, int32_t g,
        const std::vector<int32_t>& rot_in,
        const std::vector<int32_t>& rot_out);

    // Hoisted BSGS helper functions
    phantom::util::cuda_auto_ptr<uint64_t> fast_rotation_precompute(
        const PhantomContext& context, const PhantomCiphertext& ct);

    PhantomCiphertext fast_rotation_ext(
        const PhantomContext& context, const PhantomCiphertext& ct,
        int32_t rot_step, phantom::util::cuda_auto_ptr<uint64_t>& digits,
        bool add_first);

    PhantomCiphertext key_switch_ext(
        const PhantomContext& context, const PhantomCiphertext& ct);

    PhantomCiphertext key_switch_down(
        const PhantomContext& context, const PhantomCiphertext& ct);

    PhantomCiphertext key_switch_down_first(
        const PhantomContext& context, const PhantomCiphertext& ct);

    void mult_ext_inplace(
        const PhantomContext& context, PhantomCiphertext& ct,
        const PhantomPlaintext& pt);

    void add_ext_inplace(
        const PhantomContext& context, PhantomCiphertext& ct1,
        const PhantomCiphertext& ct2);

    // host-side FFT precomputation
    static std::vector<uint32_t> select_layers(uint32_t log_slots, uint32_t budget);
    static std::vector<int32_t> get_collapsed_fft_params(uint32_t slots, uint32_t level_budget, uint32_t dim1 = 0);
    static uint32_t reduce_rotation(int32_t index, uint32_t slots);

    static std::vector<std::vector<std::complex<double>>>
    coeff_encoding_one_level(const std::vector<std::complex<double>>& pows,
                             const std::vector<uint32_t>& rot_group, bool flag_i);

    static std::vector<std::vector<std::complex<double>>>
    coeff_decoding_one_level(const std::vector<std::complex<double>>& pows,
                             const std::vector<uint32_t>& rot_group, bool flag_i);

    static std::vector<std::vector<std::vector<std::complex<double>>>>
    coeff_encoding_collapse(const std::vector<std::complex<double>>& pows,
                            const std::vector<uint32_t>& rot_group,
                            uint32_t level_budget, bool flag_i);

    static std::vector<std::vector<std::vector<std::complex<double>>>>
    coeff_decoding_collapse(const std::vector<std::complex<double>>& pows,
                            const std::vector<uint32_t>& rot_group,
                            uint32_t level_budget, bool flag_i);

    std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>>
    precompute_c2s(const PhantomContext& context,
                   const std::vector<std::complex<double>>& ksi_pows,
                   const std::vector<uint32_t>& rot_group,
                   double scale, uint32_t level);

    std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>>
    precompute_s2c(const PhantomContext& context,
                   const std::vector<std::complex<double>>& ksi_pows,
                   const std::vector<uint32_t>& rot_group,
                   double scale, uint32_t level);

    PhantomCKKSEncoder& encoder_;
    PhantomRelinKey relin_key_;
    PhantomGaloisKey galois_key_;

    uint32_t correction_factor_ = 0;
    std::map<uint32_t, std::shared_ptr<BootPrecom>> precom_map_;

    static constexpr uint32_t K_UNIFORM = 512;
    static constexpr uint32_t R_UNIFORM = 6;

    static const std::vector<double> chebyshev_coeffs_;
};

} // namespace phantom
