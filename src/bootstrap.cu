#include "bootstrap.cuh"
#include <algorithm>
#include <numeric>
#include <iostream>

using namespace std;
using namespace phantom::util;
using namespace phantom::arith;

namespace phantom {

//==============================================================================
// Utility: compute automorphism index for rotation in 2n-th cyclotomic
//==============================================================================
static uint32_t FindAutomorphismIndex2nComplex(int32_t i, uint32_t m) {
    if (i == 0) return 1;
    if (i == int32_t(m - 1)) return uint32_t(i); // conjugation

    uint64_t g0;
    if (i < 0) {
        if (!phantom::arith::try_invert_uint_mod(5, m, g0))
            throw std::logic_error("FindAutomorphismIndex2nComplex: invert failed");
    } else {
        g0 = 5;
    }
    uint32_t i_unsigned = (uint32_t)std::abs(i);
    uint64_t g = g0;
    for (size_t j = 1; j < i_unsigned; j++)
        g = (g * g0) % m;
    return uint32_t(g);
}

//==============================================================================
// GPU kernel: precompute automorphism permutation map
//==============================================================================
static __device__ inline uint32_t device_reverse_bits(uint32_t operand, int bit_count) {
    return (bit_count == 0) ? 0u : (__brev(operand) >> (32 - bit_count));
}

static __global__ void PrecomputeAutoMapKernel(uint32_t n, uint32_t k, uint32_t* precomp) {
    uint32_t m    = n << 1;
    uint32_t logm = __float2uint_rn(log2f((float)m));
    uint32_t logn = __float2uint_rn(log2f((float)n));

    for (uint32_t j = threadIdx.x + blockIdx.x * blockDim.x;
         j < n;
         j += gridDim.x * blockDim.x) {
        uint32_t jTmp    = ((j << 1) + 1);
        uint32_t idx     = ((jTmp * k) - (((jTmp * k) >> logm) << logm)) >> 1;
        uint32_t jrev    = device_reverse_bits(j, logn);
        uint32_t idxrev  = device_reverse_bits(idx, logn);
        precomp[jrev] = idxrev;
    }
}

// Chebyshev coefficients for uniform secret case (degree 88)
const std::vector<double> CKKSBootstrapper::chebyshev_coeffs_ = {
    0.15421426400235561,    -0.0037671538417132409,  0.16032011744533031,      -0.0034539657223742453,
    0.17711481926851286,    -0.0027619720033372291,  0.19949802549604084,      -0.0015928034845171929,
    0.21756948616367638,    0.00010729951647566607,  0.21600427371240055,      0.0022171399198851363,
    0.17647500259573556,    0.0042856217194480991,   0.086174491919472254,     0.0054640252312780444,
    -0.046667988130649173,  0.0047346914623733714,   -0.17712686172280406,     0.0016205080004247200,
    -0.22703114241338604,   -0.0028145845916205865,  -0.13123089730288540,     -0.0056345646688793190,
    0.078818395388692147,   -0.0037868875028868542,  0.23226434602675575,      0.0021116338645426574,
    0.13985510526186795,    0.0059365649669377071,   -0.13918475289368595,     0.0018580676740836374,
    -0.23254376365752788,   -0.0054103844866927788,  0.056840618403875359,     -0.0035227192748552472,
    0.25667909012207590,    0.0055029673963982112,   -0.073334392714092062,    0.0027810273357488265,
    -0.24912792167850559,   -0.0069524866497120566,  0.21288810409948347,      0.0017810057298691725,
    0.088760951809475269,   0.0055957188940032095,   -0.31937177676259115,     -0.0087539416335935556,
    0.34748800245527145,    0.0075378299617709235,   -0.25116537379803394,     -0.0047285674679876204,
    0.13970502851683486,    0.0023672533925155220,   -0.063649401080083698,    -0.00098993213448982727,
    0.024597838934816905,   0.00035553235917057483,  -0.0082485030307578155,   -0.00011176184313622549,
    0.0024390574829093264,  0.000031180384864488629, -0.00064373524734389861,  -7.8036008952377965e-6,
    0.00015310015145922058, 1.7670804180220134e-6,   -0.000033066844379476900, -3.6460909134279425e-7,
    6.5276969021754105e-6,  6.8957843666189918e-8,   -1.1842811187642386e-6,   -1.2015133285307312e-8,
    1.9839339947648331e-7,  1.9372045971100854e-9,   -3.0815418032523593e-8,   -2.9013806338735810e-10,
    4.4540904298173700e-9,  4.0505136697916078e-11,  -6.0104912807134771e-10,  -5.2873323696828491e-12,
    7.5943206779351725e-11, 6.4679566322060472e-13,  -9.0081200925539902e-12,  -7.4396949275292252e-14,
    1.0057423059167244e-12, 8.1701187638005194e-15,  -1.0611736208855373e-13,  -8.9597492970451533e-16,
    1.1421575296031385e-14
};

//==============================================================================
// Static utility functions
//==============================================================================

uint32_t CKKSBootstrapper::reduce_rotation(int32_t index, uint32_t slots) {
    int32_t islots = int32_t(slots);
    if ((slots & (slots - 1)) == 0) {
        int32_t n = std::log2(slots);
        if (index >= 0)
            return index - ((index >> n) << n);
        return index + islots + ((int32_t(std::fabs(index)) >> n) << n);
    }
    return (islots + index % islots) % islots;
}

std::vector<uint32_t> CKKSBootstrapper::select_layers(uint32_t log_slots, uint32_t budget) {
    uint32_t layers = std::ceil(static_cast<double>(log_slots) / budget);
    uint32_t rows = log_slots / layers;
    uint32_t rem = log_slots % layers;

    uint32_t dim = rows;
    if (rem != 0) dim = rows + 1;

    if (dim < budget) {
        layers -= 1;
        rows = log_slots / layers;
        rem = log_slots - rows * layers;
        dim = rows;
        if (rem != 0) dim = rows + 1;

        while (dim != budget) {
            rows -= 1;
            rem = log_slots - rows * layers;
            dim = rows;
            if (rem != 0) dim = rows + 1;
        }
    }

    return {layers, rows, rem};
}

std::vector<int32_t> CKKSBootstrapper::get_collapsed_fft_params(
    uint32_t slots, uint32_t level_budget, uint32_t dim1) {

    uint32_t log_slots = std::log2(slots);
    if (log_slots == 0) log_slots = 1;

    auto dims = select_layers(log_slots, level_budget);
    int32_t layers_coll = dims[0];
    int32_t rem_coll = dims[2];

    bool flag_rem = (rem_coll != 0);
    uint32_t num_rot = (1 << (layers_coll + 1)) - 1;
    uint32_t num_rot_rem = (1 << (rem_coll + 1)) - 1;

    int32_t g;
    if (dim1 == 0 || dim1 > num_rot) {
        if (num_rot > 7)
            g = (1 << (int32_t(layers_coll / 2) + 2));
        else
            g = (1 << (int32_t(layers_coll / 2) + 1));
    } else {
        g = dim1;
    }
    int32_t b = (num_rot + 1) / g;

    int32_t b_rem = 0, g_rem = 0;
    if (flag_rem) {
        if (num_rot_rem > 7)
            g_rem = (1 << (int32_t(rem_coll / 2) + 2));
        else
            g_rem = (1 << (int32_t(rem_coll / 2) + 1));
        b_rem = (num_rot_rem + 1) / g_rem;
    }

    return {int32_t(level_budget), layers_coll, rem_coll, int32_t(num_rot), b, g,
            int32_t(num_rot_rem), b_rem, g_rem};
}

std::vector<uint32_t> CKKSBootstrapper::get_galois_elements(
    uint32_t poly_degree, uint32_t num_slots,
    const std::vector<uint32_t>& level_budget) {

    uint32_t N = poly_degree;
    uint32_t M = 2 * N;
    uint32_t slots = (num_slots == 0) ? N / 2 : num_slots;

    auto params_enc = get_collapsed_fft_params(slots, level_budget[0]);
    auto params_dec = get_collapsed_fft_params(slots, level_budget[1]);

    std::vector<int32_t> steps;

    // encoding rotations
    {
        int32_t budget = params_enc[boot_params::LEVEL_BUDGET];
        int32_t layers = params_enc[boot_params::LAYERS_COLL];
        int32_t rem = params_enc[boot_params::LAYERS_REM];
        int32_t num_rot = params_enc[boot_params::NUM_ROTATIONS];
        int32_t b = params_enc[boot_params::BABY_STEP];
        int32_t g = params_enc[boot_params::GIANT_STEP];
        int32_t num_rot_rem = params_enc[boot_params::NUM_ROTATIONS_REM];
        int32_t b_rem = params_enc[boot_params::BABY_STEP_REM];
        int32_t g_rem = params_enc[boot_params::GIANT_STEP_REM];
        int32_t flag_rem = (rem != 0) ? 1 : 0;
        int32_t stop = (rem == 0) ? -1 : 0;

        for (int32_t si = budget - 1; si > stop; si--) {
            for (int32_t j = 0; j < g; j++) {
                int32_t rot = reduce_rotation(
                    (j - int32_t((num_rot + 1) / 2) + 1) *
                    (1 << ((si - flag_rem) * layers + rem)), slots);
                if (rot != 0) steps.push_back(rot);
            }
            for (int32_t i = 1; i < b; i++) {
                int32_t rot = reduce_rotation(
                    (g * i) * (1 << ((si - flag_rem) * layers + rem)), M / 4);
                if (rot != 0) steps.push_back(rot);
            }
        }
        if (flag_rem) {
            for (int32_t j = 0; j < g_rem; j++) {
                int32_t rot = reduce_rotation(
                    (j - int32_t((num_rot_rem + 1) / 2) + 1), slots);
                if (rot != 0) steps.push_back(rot);
            }
            for (int32_t i = 1; i < b_rem; i++) {
                int32_t rot = reduce_rotation(g_rem * i, M / 4);
                if (rot != 0) steps.push_back(rot);
            }
        }
    }

    // decoding rotations
    {
        int32_t budget = params_dec[boot_params::LEVEL_BUDGET];
        int32_t layers = params_dec[boot_params::LAYERS_COLL];
        int32_t rem = params_dec[boot_params::LAYERS_REM];
        int32_t num_rot = params_dec[boot_params::NUM_ROTATIONS];
        int32_t b = params_dec[boot_params::BABY_STEP];
        int32_t g = params_dec[boot_params::GIANT_STEP];
        int32_t num_rot_rem = params_dec[boot_params::NUM_ROTATIONS_REM];
        int32_t b_rem = params_dec[boot_params::BABY_STEP_REM];
        int32_t g_rem = params_dec[boot_params::GIANT_STEP_REM];
        int32_t flag_rem = (rem != 0) ? 1 : 0;

        for (int32_t si = 0; si < budget - flag_rem; si++) {
            for (int32_t j = 0; j < g; j++) {
                int32_t rot = reduce_rotation(
                    (j - int32_t((num_rot + 1) / 2) + 1) *
                    (1 << (si * layers)), M / 4);
                if (rot != 0) steps.push_back(rot);
            }
            for (int32_t i = 1; i < b; i++) {
                int32_t rot = reduce_rotation(
                    (g * i) * (1 << (si * layers)), M / 4);
                if (rot != 0) steps.push_back(rot);
            }
        }
        if (flag_rem) {
            int32_t si = budget - flag_rem;
            for (int32_t j = 0; j < g_rem; j++) {
                int32_t rot = reduce_rotation(
                    (j - int32_t((num_rot_rem + 1) / 2) + 1) *
                    (1 << (si * layers)), M / 4);
                if (rot != 0) steps.push_back(rot);
            }
            for (int32_t i = 1; i < b_rem; i++) {
                int32_t rot = reduce_rotation(
                    (g_rem * i) * (1 << (si * layers)), M / 4);
                if (rot != 0) steps.push_back(rot);
            }
        }
    }

    // sparse packing rotations
    if (slots < N / 2) {
        for (uint32_t j = 1; j < N / (2 * slots); j <<= 1)
            steps.push_back(j * slots);
    }

    std::sort(steps.begin(), steps.end());
    steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
    steps.erase(std::remove(steps.begin(), steps.end(), 0), steps.end());

    // convert steps to galois elements (get_elt_from_step expects coeff_count = N, not M)
    std::vector<uint32_t> elts;
    for (auto s : steps) {
        uint32_t elt = phantom::util::get_elt_from_step(s, N);
        elts.push_back(elt);
    }
    // conjugation element
    elts.push_back(M - 1);

    std::sort(elts.begin(), elts.end());
    elts.erase(std::unique(elts.begin(), elts.end()), elts.end());

    return elts;
}

uint32_t CKKSBootstrapper::get_bootstrap_depth(const std::vector<uint32_t>& level_budget) {
    // Chebyshev evaluation (degree 88, binary tree): ceil(log2(88))+1 = 8 levels
    // Double angle: R_UNIFORM = 6 levels
    // Overhead: 1 mult_by_const rescale + (budget-1) C2S internal + 1 post-C2S rescale
    uint32_t approx_mod_depth = 14;  // 8 (chebyshev) + 6 (double angle)
    return approx_mod_depth + level_budget[0] + level_budget[1] + 2;
}

//==============================================================================
// FFT coefficient computation (host-side)
//==============================================================================

std::vector<std::vector<std::complex<double>>>
CKKSBootstrapper::coeff_encoding_one_level(
    const std::vector<std::complex<double>>& pows,
    const std::vector<uint32_t>& rot_group, bool flag_i) {

    uint32_t dim = pows.size() - 1;
    uint32_t slots = rot_group.size();
    uint32_t log_slots = std::log2(slots);

    std::vector<std::vector<std::complex<double>>> coeff(3 * log_slots);
    for (uint32_t i = 0; i < 3 * log_slots; i++)
        coeff[i] = std::vector<std::complex<double>>(slots, {0, 0});

    const std::complex<double> I(0, 1);

    for (uint32_t m = slots; m > 1; m >>= 1) {
        uint32_t s = std::log2(m) - 1;
        for (uint32_t k = 0; k < slots; k += m) {
            uint32_t lenh = m >> 1;
            uint32_t lenq = m << 2;
            for (uint32_t j = 0; j < lenh; j++) {
                uint32_t jTwiddle = (lenq - (rot_group[j] % lenq)) * (dim / lenq);
                if (flag_i && (m == 2)) {
                    std::complex<double> w = std::exp(-M_PI / 2.0 * I) * pows[jTwiddle];
                    coeff[s + log_slots][j + k] = std::exp(-M_PI / 2.0 * I);
                    coeff[s + 2 * log_slots][j + k] = std::exp(-M_PI / 2.0 * I);
                    coeff[s + log_slots][j + k + lenh] = -w;
                    coeff[s][j + k + lenh] = w;
                } else {
                    std::complex<double> w = pows[jTwiddle];
                    coeff[s + log_slots][j + k] = 1;
                    coeff[s + 2 * log_slots][j + k] = 1;
                    coeff[s + log_slots][j + k + lenh] = -w;
                    coeff[s][j + k + lenh] = w;
                }
            }
        }
    }
    return coeff;
}

std::vector<std::vector<std::complex<double>>>
CKKSBootstrapper::coeff_decoding_one_level(
    const std::vector<std::complex<double>>& pows,
    const std::vector<uint32_t>& rot_group, bool flag_i) {

    uint32_t dim = pows.size() - 1;
    uint32_t slots = rot_group.size();
    uint32_t log_slots = std::log2(slots);

    std::vector<std::vector<std::complex<double>>> coeff(3 * log_slots);
    for (uint32_t i = 0; i < 3 * log_slots; i++)
        coeff[i] = std::vector<std::complex<double>>(slots, {0, 0});

    const std::complex<double> I(0, 1);

    for (uint32_t m = 2; m <= slots; m <<= 1) {
        uint32_t s = std::log2(m) - 1;
        for (uint32_t k = 0; k < slots; k += m) {
            uint32_t lenh = m >> 1;
            uint32_t lenq = m << 2;
            for (uint32_t j = 0; j < lenh; j++) {
                uint32_t jTwiddle = (rot_group[j] % lenq) * (dim / lenq);
                if (flag_i && (m == 2)) {
                    std::complex<double> w = std::exp(M_PI / 2.0 * I) * pows[jTwiddle];
                    coeff[s + log_slots][j + k] = std::exp(M_PI / 2.0 * I);
                    coeff[s + 2 * log_slots][j + k] = w;
                    coeff[s + log_slots][j + k + lenh] = -w;
                    coeff[s][j + k + lenh] = std::exp(M_PI / 2.0 * I);
                } else {
                    std::complex<double> w = pows[jTwiddle];
                    coeff[s + log_slots][j + k] = 1;
                    coeff[s + 2 * log_slots][j + k] = w;
                    coeff[s + log_slots][j + k + lenh] = -w;
                    coeff[s][j + k + lenh] = 1;
                }
            }
        }
    }
    return coeff;
}

std::vector<std::vector<std::vector<std::complex<double>>>>
CKKSBootstrapper::coeff_encoding_collapse(
    const std::vector<std::complex<double>>& pows,
    const std::vector<uint32_t>& rot_group,
    uint32_t level_budget, bool flag_i) {

    uint32_t slots = rot_group.size();
    uint32_t log_slots = std::log2(slots);
    auto dims = select_layers(log_slots, level_budget);
    int32_t layers_coll = dims[0];
    int32_t rem_coll = dims[2];

    int32_t dim_collapse = int32_t(level_budget);
    int32_t stop = (rem_coll == 0) ? -1 : 0;
    int32_t flag_rem = (rem_coll == 0) ? 0 : 1;

    uint32_t num_rot = (1 << (layers_coll + 1)) - 1;
    uint32_t num_rot_rem = (1 << (rem_coll + 1)) - 1;

    auto coeff1 = coeff_encoding_one_level(pows, rot_group, flag_i);

    std::vector<std::vector<std::vector<std::complex<double>>>> coeff(dim_collapse);
    for (uint32_t i = 0; i < uint32_t(dim_collapse); i++) {
        uint32_t nr = (flag_rem && i == 0) ? num_rot_rem : num_rot;
        coeff[i].resize(nr);
        for (uint32_t j = 0; j < nr; j++)
            coeff[i][j].resize(slots, {0, 0});
    }

    for (int32_t s = dim_collapse - 1; s > stop; s--) {
        int32_t top = int32_t(log_slots) - (dim_collapse - 1 - s) * layers_coll - 1;
        for (int32_t l = 0; l < layers_coll; l++) {
            if (l == 0) {
                coeff[s][0] = coeff1[top];
                coeff[s][1] = coeff1[top + log_slots];
                coeff[s][2] = coeff1[top + 2 * log_slots];
            } else {
                auto temp = coeff[s];
                for (auto& v : coeff[s])
                    std::fill(v.begin(), v.end(), std::complex<double>(0, 0));
                uint32_t t = 0;
                for (int32_t u = 0; u < (1 << (l + 1)) - 1; u++) {
                    for (uint32_t k = 0; k < slots; k++) {
                        coeff[s][u + t][k] += coeff1[top - l][k] *
                            temp[u][reduce_rotation(k - (1 << (top - l)), slots)];
                        coeff[s][u + t + 1][k] += coeff1[top - l + log_slots][k] * temp[u][k];
                        coeff[s][u + t + 2][k] += coeff1[top - l + 2 * log_slots][k] *
                            temp[u][reduce_rotation(k + (1 << (top - l)), slots)];
                    }
                    t += 1;
                }
            }
        }
    }

    if (flag_rem) {
        int32_t s = 0;
        int32_t top = int32_t(log_slots) - (dim_collapse - 1 - s) * layers_coll - 1;
        for (int32_t l = 0; l < rem_coll; l++) {
            if (l == 0) {
                coeff[s][0] = coeff1[top];
                coeff[s][1] = coeff1[top + log_slots];
                coeff[s][2] = coeff1[top + 2 * log_slots];
            } else {
                auto temp = coeff[s];
                for (auto& v : coeff[s])
                    std::fill(v.begin(), v.end(), std::complex<double>(0, 0));
                uint32_t t = 0;
                for (int32_t u = 0; u < (1 << (l + 1)) - 1; u++) {
                    for (uint32_t k = 0; k < slots; k++) {
                        coeff[s][u + t][k] += coeff1[top - l][k] *
                            temp[u][reduce_rotation(k - (1 << (top - l)), slots)];
                        coeff[s][u + t + 1][k] += coeff1[top - l + log_slots][k] * temp[u][k];
                        coeff[s][u + t + 2][k] += coeff1[top - l + 2 * log_slots][k] *
                            temp[u][reduce_rotation(k + (1 << (top - l)), slots)];
                    }
                    t += 1;
                }
            }
        }
    }

    return coeff;
}

std::vector<std::vector<std::vector<std::complex<double>>>>
CKKSBootstrapper::coeff_decoding_collapse(
    const std::vector<std::complex<double>>& pows,
    const std::vector<uint32_t>& rot_group,
    uint32_t level_budget, bool flag_i) {

    uint32_t slots = rot_group.size();
    uint32_t log_slots = std::log2(slots);
    auto dims = select_layers(log_slots, level_budget);
    int32_t layers_coll = dims[0];
    int32_t rows_coll = dims[1];
    int32_t rem_coll = dims[2];

    int32_t dim_collapse = int32_t(level_budget);
    int32_t flag_rem = (rem_coll != 0) ? 1 : 0;

    uint32_t num_rot = (1 << (layers_coll + 1)) - 1;
    uint32_t num_rot_rem = (1 << (rem_coll + 1)) - 1;

    auto coeff1 = coeff_decoding_one_level(pows, rot_group, flag_i);

    std::vector<std::vector<std::vector<std::complex<double>>>> coeff(dim_collapse);
    for (uint32_t i = 0; i < uint32_t(dim_collapse); i++) {
        uint32_t nr = (flag_rem && i == uint32_t(dim_collapse - 1)) ? num_rot_rem : num_rot;
        coeff[i].resize(nr);
        for (uint32_t j = 0; j < nr; j++)
            coeff[i][j].resize(slots, {0, 0});
    }

    for (int32_t s = 0; s < rows_coll; s++) {
        for (int32_t l = 0; l < layers_coll; l++) {
            if (l == 0) {
                coeff[s][0] = coeff1[s * layers_coll];
                coeff[s][1] = coeff1[log_slots + s * layers_coll];
                coeff[s][2] = coeff1[2 * log_slots + s * layers_coll];
            } else {
                auto temp = coeff[s];
                for (auto& v : coeff[s])
                    std::fill(v.begin(), v.end(), std::complex<double>(0, 0));
                for (uint32_t t = 0; t < 3; t++) {
                    for (int32_t u = 0; u < (1 << (l + 1)) - 1; u++) {
                        for (uint32_t k = 0; k < slots; k++) {
                            if (t == 0)
                                coeff[s][u][k] += coeff1[s * layers_coll + l][k] * temp[u][k];
                            if (t == 1)
                                coeff[s][u + (1 << l)][k] +=
                                    coeff1[s * layers_coll + l + log_slots][k] * temp[u][k];
                            if (t == 2)
                                coeff[s][u + (1 << (l + 1))][k] +=
                                    coeff1[s * layers_coll + l + 2 * log_slots][k] * temp[u][k];
                        }
                    }
                }
            }
        }
    }

    if (flag_rem) {
        int32_t s = rows_coll;
        for (int32_t l = 0; l < rem_coll; l++) {
            if (l == 0) {
                coeff[s][0] = coeff1[s * layers_coll];
                coeff[s][1] = coeff1[log_slots + s * layers_coll];
                coeff[s][2] = coeff1[2 * log_slots + s * layers_coll];
            } else {
                auto temp = coeff[s];
                for (auto& v : coeff[s])
                    std::fill(v.begin(), v.end(), std::complex<double>(0, 0));
                for (uint32_t t = 0; t < 3; t++) {
                    for (int32_t u = 0; u < (1 << (l + 1)) - 1; u++) {
                        for (uint32_t k = 0; k < slots; k++) {
                            if (t == 0)
                                coeff[s][u][k] += coeff1[s * layers_coll + l][k] * temp[u][k];
                            if (t == 1)
                                coeff[s][u + (1 << l)][k] +=
                                    coeff1[s * layers_coll + l + log_slots][k] * temp[u][k];
                            if (t == 2)
                                coeff[s][u + (1 << (l + 1))][k] +=
                                    coeff1[s * layers_coll + l + 2 * log_slots][k] * temp[u][k];
                        }
                    }
                }
            }
        }
    }

    return coeff;
}

//==============================================================================
// CUDA kernels for bootstrap operations
//==============================================================================

__global__ void switch_modulus_kernel(
    const uint64_t* src, uint64_t* dst,
    const DModulus* modulus, uint32_t N, uint32_t num_moduli) {
    // src has N coefficients in coefficient form (single RNS basis, mod q0)
    // dst has num_moduli * N slots, first N already filled
    // fill dst[i*N + j] = signed_lift(src[j], q0) mod modulus[i] for i >= 1
    //
    // Signed lift: values in [q0/2, q0) are negative
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * (num_moduli - 1);
    if (tid >= total) return;

    uint32_t mod_idx = tid / N + 1;
    uint32_t coeff_idx = tid % N;
    uint64_t val = src[coeff_idx];
    uint64_t q0 = modulus[0].value();
    uint64_t qi = modulus[mod_idx].value();

    if (val > q0 / 2) {
        // Negative coefficient: actual value is val - q0 (negative)
        uint64_t neg_val = q0 - val;  // |actual value|
        uint64_t r = neg_val % qi;
        dst[mod_idx * N + coeff_idx] = (r == 0) ? 0 : (qi - r);
    } else {
        dst[mod_idx * N + coeff_idx] = val % qi;
    }
}

__global__ void init_monomial_kernel(
    uint64_t* out, uint32_t index, uint32_t power_reduced,
    uint32_t total, uint32_t N, const DModulus* modulus) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    uint32_t mod_idx = tid / N;
    uint32_t coeff_idx = tid % N;
    uint64_t qi = modulus[mod_idx].value();

    if (coeff_idx == index) {
        out[tid] = (power_reduced >= N) ? (qi - 1) : 1;
    } else {
        out[tid] = 0;
    }
}

__global__ void multiply_scalar_kernel(
    uint64_t* data, uint64_t scalar,
    const DModulus* modulus, uint32_t N, uint32_t num_moduli) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * num_moduli;
    if (tid >= total) return;

    uint32_t mod_idx = tid / N;
    uint64_t qi = modulus[mod_idx].value();
    uint64_t s = scalar % qi;
    unsigned __int128 prod = (unsigned __int128)data[tid] * s;
    data[tid] = (uint64_t)(prod % qi);
}

__global__ void multiply_poly_kernel(
    uint64_t* dst, const uint64_t* src,
    const DModulus* modulus, uint32_t N, uint32_t num_moduli) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * num_moduli;
    if (tid >= total) return;

    uint32_t mod_idx = tid / N;
    uint64_t qi = modulus[mod_idx].value();
    unsigned __int128 prod = (unsigned __int128)dst[tid] * src[tid];
    dst[tid] = (uint64_t)(prod % qi);
}

__global__ void multiply_per_mod_scalar_kernel(
    uint64_t* data, const uint64_t* scalars,
    const DModulus* modulus, uint32_t N, uint32_t num_moduli) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * num_moduli;
    if (tid >= total) return;

    uint32_t mod_idx = tid / N;
    uint64_t qi = modulus[mod_idx].value();
    uint64_t s = scalars[mod_idx];
    unsigned __int128 prod = (unsigned __int128)data[tid] * s;
    data[tid] = (uint64_t)(prod % qi);
}

__global__ void add_scalar_to_first_kernel(
    uint64_t* data, const uint64_t* scalars,
    const DModulus* modulus, uint32_t N, uint32_t num_moduli) {
    uint32_t mod_idx = blockIdx.x;
    if (mod_idx >= num_moduli) return;

    uint64_t qi = modulus[mod_idx].value();
    uint64_t val = data[mod_idx * N] + scalars[mod_idx];
    data[mod_idx * N] = (val >= qi) ? (val - qi) : val;
}

__global__ void add_const_ntt_kernel(
    uint64_t* data, const uint64_t* scalars,
    const DModulus* modulus, uint32_t N, uint32_t num_moduli) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * num_moduli;
    if (tid >= total) return;

    uint32_t mod_idx = tid / N;
    uint64_t qi = modulus[mod_idx].value();
    uint64_t val = data[tid] + scalars[mod_idx];
    data[tid] = (val >= qi) ? (val - qi) : val;
}

//==============================================================================
// Scalar ciphertext operations
//==============================================================================

void CKKSBootstrapper::mult_by_const(
    const PhantomContext& context, PhantomCiphertext& ct, double scalar) {

    if (ct.GetNoiseScaleDeg() == 2)
        rescale_to_next_inplace(context, ct);

    auto& ctx = context.get_context_data(ct.chain_index());
    auto& parms = ctx.parms();
    auto& coeff_mod = parms.coeff_modulus();
    uint32_t N = parms.poly_modulus_degree();
    uint32_t coeff_mod_size = coeff_mod.size();

    // scFactor = last prime at current chain level (the rescaling divisor)
    double scFactor = static_cast<double>(coeff_mod[coeff_mod_size - 1].value());

    // Use 128-bit integer arithmetic to compute round(scalar * scFactor)
    // and reduce mod each prime — avoids CKKS encoding noise entirely.
    typedef __int128 DoubleInteger;
    const int32_t MAX_BITS = 125;

    int32_t logApprox = 0;
    double res = std::fabs(scalar * scFactor);
    if (res > 0) {
        int32_t logSF = static_cast<int32_t>(std::ceil(std::log2(res)));
        int32_t logValid = (logSF <= MAX_BITS) ? logSF : MAX_BITS;
        logApprox = logSF - logValid;
    }
    double approxFactor = std::pow(2.0, logApprox);

    DoubleInteger large = static_cast<DoubleInteger>(scalar / approxFactor * scFactor + 0.5);
    if (scalar < 0)
        large = static_cast<DoubleInteger>(scalar / approxFactor * scFactor - 0.5);
    DoubleInteger large_abs = (large < 0) ? -large : large;
    DoubleInteger bound = static_cast<DoubleInteger>(1) << 63;

    std::vector<uint64_t> factors(coeff_mod_size);

    if (large_abs >= bound) {
        for (uint32_t i = 0; i < coeff_mod_size; i++) {
            DoubleInteger qi = static_cast<DoubleInteger>(coeff_mod[i].value());
            DoubleInteger reduced = large % qi;
            factors[i] = (reduced < 0)
                ? static_cast<uint64_t>(reduced + qi)
                : static_cast<uint64_t>(reduced);
        }
    } else {
        int64_t scConstant = static_cast<int64_t>(large);
        for (uint32_t i = 0; i < coeff_mod_size; i++) {
            int64_t qi = static_cast<int64_t>(coeff_mod[i].value());
            int64_t reduced = scConstant % qi;
            factors[i] = (reduced < 0) ? reduced + qi : reduced;
        }
    }

    // Scale back by approxFactor in RNS
    if (logApprox > 0) {
        int32_t remaining = logApprox;
        while (remaining > 0) {
            int32_t step = std::min(remaining, 60);
            uint64_t intStep = uint64_t(1) << step;
            for (uint32_t i = 0; i < coeff_mod_size; i++) {
                unsigned __int128 prod = (unsigned __int128)factors[i] * intStep;
                factors[i] = static_cast<uint64_t>(prod % coeff_mod[i].value());
            }
            remaining -= step;
        }
    }

    uint64_t* d_scalars;
    cudaMalloc(&d_scalars, coeff_mod_size * sizeof(uint64_t));
    cudaMemcpy(d_scalars, factors.data(), coeff_mod_size * sizeof(uint64_t),
               cudaMemcpyHostToDevice);

    auto base_rns = context.gpu_rns_tables().modulus();
    uint32_t rns_coeff_count = N * coeff_mod_size;
    const auto& s = cudaStreamPerThread;
    uint32_t block = 256;
    uint32_t grid = (rns_coeff_count + block - 1) / block;

    for (size_t i = 0; i < ct.size(); i++) {
        multiply_per_mod_scalar_kernel<<<grid, block, 0, s>>>(
            ct.data() + i * rns_coeff_count, d_scalars,
            base_rns, N, coeff_mod_size);
    }
    cudaStreamSynchronize(s);
    cudaFree(d_scalars);

    ct.SetNoiseScaleDeg(ct.GetNoiseScaleDeg() + 1);
    ct.set_scale(ct.scale() * scFactor);
}

void CKKSBootstrapper::mult_by_integer(
    const PhantomContext& context, PhantomCiphertext& ct, uint64_t integer) {

    auto& ctx = context.get_context_data(ct.chain_index());
    auto& parms = ctx.parms();
    uint32_t N = parms.poly_modulus_degree();
    uint32_t coeff_mod_size = parms.coeff_modulus().size();
    auto base_rns = context.gpu_rns_tables().modulus();
    uint32_t rns_coeff_count = N * coeff_mod_size;

    const auto& s = cudaStreamPerThread;
    uint32_t block = 256;

    for (size_t i = 0; i < ct.size(); i++) {
        uint32_t grid = (rns_coeff_count + block - 1) / block;
        multiply_scalar_kernel<<<grid, block, 0, s>>>(
            ct.data() + i * rns_coeff_count, integer,
            base_rns, N, coeff_mod_size);
    }
    cudaStreamSynchronize(s);
}

void CKKSBootstrapper::mult_by_monomial(
    const PhantomContext& context, PhantomCiphertext& ct, uint32_t power) {

    const auto& s = cudaStreamPerThread;
    auto& ctx = context.get_context_data(ct.chain_index());
    auto& parms = ctx.parms();
    uint32_t N = parms.poly_modulus_degree();
    uint32_t M = 2 * N;
    uint32_t coeff_mod_size = parms.coeff_modulus().size();
    auto base_rns = context.gpu_rns_tables().modulus();
    uint32_t rns_coeff_count = N * coeff_mod_size;

    uint32_t power_reduced = power % M;
    uint32_t index = power % N;

    uint32_t total = coeff_mod_size * N;
    auto monomial = phantom::util::make_cuda_auto_ptr<uint64_t>(total, s);
    uint32_t block = 256;
    uint32_t grid = (total + block - 1) / block;

    init_monomial_kernel<<<grid, block, 0, s>>>(
        monomial.get(), index, power_reduced, total, N, base_rns);

    nwt_2d_radix8_forward_inplace(monomial.get(), context.gpu_rns_tables(),
                                   coeff_mod_size, 0, s);

    for (size_t i = 0; i < ct.size(); i++) {
        grid = (rns_coeff_count + block - 1) / block;
        multiply_poly_kernel<<<grid, block, 0, s>>>(
            ct.data() + i * rns_coeff_count, monomial.get(),
            base_rns, N, coeff_mod_size);
    }
    cudaStreamSynchronize(s);
}

void CKKSBootstrapper::add_const(
    const PhantomContext& context, PhantomCiphertext& ct, double scalar) {

    if (scalar == 0.0) return;

    std::vector<cuDoubleComplex> vals(encoder_.slot_count());
    for (size_t i = 0; i < vals.size(); i++)
        vals[i] = make_cuDoubleComplex(scalar, 0.0);

    PhantomPlaintext pt;
    encoder_.encode(context, vals, ct.scale(), pt, ct.chain_index());
    add_plain_inplace(context, ct, pt);
}

PhantomCiphertext CKKSBootstrapper::conjugate(
    const PhantomContext& context, const PhantomCiphertext& ct) {

    auto& ctx = context.get_context_data(ct.chain_index());
    auto& parms = ctx.parms();
    uint32_t N = parms.poly_modulus_degree();
    uint32_t galois_elt = 2 * N - 1;
    return apply_galois(context, ct, galois_elt, galois_key_);
}

//==============================================================================
// ModRaise: raise ciphertext from current level to first level
//==============================================================================

PhantomCiphertext CKKSBootstrapper::mod_raise(
    const PhantomContext& context, PhantomCiphertext& ct) {

    const auto& s = cudaStreamPerThread;

    uint32_t src_mod_size = ct.coeff_modulus_size();
    uint32_t N = ct.poly_modulus_degree();

    auto c0 = ct.data();
    auto c1 = ct.data() + src_mod_size * N;

    // INTT to get coefficient form
    nwt_2d_radix8_backward_inplace(c0, context.gpu_rns_tables(), src_mod_size, 0, s);
    nwt_2d_radix8_backward_inplace(c1, context.gpu_rns_tables(), src_mod_size, 0, s);

    PhantomCiphertext raised;
    raised.resize(context, context.get_first_index(), ct.size(), s);
    raised.set_scale(ct.scale());
    raised.SetNoiseScaleDeg(ct.GetNoiseScaleDeg());

    uint32_t dst_mod_size = raised.coeff_modulus_size();
    uint32_t dst_rns_count = dst_mod_size * N;

    auto c0_r = raised.data();
    auto c1_r = raised.data() + dst_rns_count;

    cudaMemcpyAsync(c0_r, c0, N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(c1_r, c1, N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s);

    auto base_rns = context.gpu_rns_tables().modulus();
    uint32_t block = 256;
    uint32_t total = N * (dst_mod_size - 1);
    uint32_t grid = (total + block - 1) / block;

    switch_modulus_kernel<<<grid, block, 0, s>>>(c0, c0_r, base_rns, N, dst_mod_size);
    switch_modulus_kernel<<<grid, block, 0, s>>>(c1, c1_r, base_rns, N, dst_mod_size);

    nwt_2d_radix8_forward_inplace(c0_r, context.gpu_rns_tables(), dst_mod_size, 0, s);
    nwt_2d_radix8_forward_inplace(c1_r, context.gpu_rns_tables(), dst_mod_size, 0, s);
    raised.set_ntt_form(true);

    nwt_2d_radix8_forward_inplace(c0, context.gpu_rns_tables(), src_mod_size, 0, s);
    nwt_2d_radix8_forward_inplace(c1, context.gpu_rns_tables(), src_mod_size, 0, s);

    cudaStreamSynchronize(s);
    return raised;
}

//==============================================================================
// Setup: precompute FFT coefficients
//==============================================================================

void CKKSBootstrapper::setup(
    const PhantomContext& context,
    std::vector<uint32_t> level_budget,
    std::vector<uint32_t> dim1,
    uint32_t num_slots,
    uint32_t correction_factor) {

    auto& context_data = context.get_context_data(context.get_first_index());
    auto& parms = context_data.parms();
    size_t N = parms.poly_modulus_degree();
    int M = 2 * N;
    int slots = (num_slots == 0) ? N / 2 : num_slots;

    if (correction_factor == 0) {
        auto tmp = std::round(-0.265 * (2 * std::log2(M / 2) + std::log2(slots)) + 19.1);
        if (tmp < 7) correction_factor_ = 7;
        else if (tmp > 13) correction_factor_ = 13;
        else correction_factor_ = static_cast<int>(tmp);
    } else {
        correction_factor_ = correction_factor;
    }

    if (level_budget[0] > uint32_t(std::log2(slots)))
        level_budget[0] = std::log2(slots);
    if (level_budget[0] < 1) level_budget[0] = 1;
    if (level_budget[1] > uint32_t(std::log2(slots)))
        level_budget[1] = std::log2(slots);
    if (level_budget[1] < 1) level_budget[1] = 1;

    precom_map_[slots] = std::make_shared<BootPrecom>();
    auto precom = precom_map_[slots];
    precom->slots = slots;
    precom->dim1 = dim1[0];
    precom->params_enc = get_collapsed_fft_params(slots, level_budget[0], dim1[0]);
    precom->params_dec = get_collapsed_fft_params(slots, level_budget[1], dim1[1]);

    uint32_t m = 4 * slots;
    std::vector<uint32_t> rot_group(slots);
    uint32_t five_pows = 1;
    for (uint32_t i = 0; i < uint32_t(slots); i++) {
        rot_group[i] = five_pows;
        five_pows *= 5;
        five_pows %= m;
    }

    std::vector<std::complex<double>> ksi_pows(m + 1);
    for (uint32_t j = 0; j < m; j++) {
        double angle = 2.0 * M_PI * j / m;
        ksi_pows[j] = {cos(angle), sin(angle)};
    }
    ksi_pows[m] = ksi_pows[0];

    double q_double = static_cast<double>(parms.coeff_modulus()[0].value());
    unsigned __int128 factor = (unsigned __int128)1 << ((uint32_t)std::round(std::log2(q_double)));
    double pre = q_double / factor;
    double scale_enc = pre;
    double scale_dec = 1.0 / pre;

    uint32_t approx_mod_depth = 14;  // 8 (chebyshev) + 6 (double angle)
    uint32_t enc_budget = precom->params_enc[boot_params::LEVEL_BUDGET];
    uint32_t dec_budget = precom->params_dec[boot_params::LEVEL_BUDGET];

    auto& key_parms = context.get_context_data(0).parms();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_parms.coeff_modulus().size();
    uint32_t L0 = size_QP - size_P;

    // Chain index where C2S plaintexts are encoded
    // mod_raise outputs at chain 1 (get_first_index) + two Scale rescales → chain 3.
    // C2S processes layers from si=levelBudget-1 (first) to si=0 (last).
    // precompute_c2s encodes layer si at chain (l_enc - si).
    // The FIRST processed layer (si=enc_budget-1) must match ciphertext chain 3:
    //   l_enc - (enc_budget - 1) = 3  =>  l_enc = enc_budget + 2
    uint32_t l_enc = enc_budget + 2;

    // Chain index where S2C plaintexts are encoded
    // After Scale (2) + C2S (enc_budget) + post-C2S (1) + EvalMod (14)
    // = enc_budget + approx_mod_depth + 3
    // precompute_s2c encodes layer si at chain (l_dec + si).
    // S2C first layer (si=0) must match ciphertext at entry chain.
    uint32_t l_dec = std::min(enc_budget + approx_mod_depth + 3, L0 - dec_budget);

    precom->enc_precom = precompute_c2s(context, ksi_pows, rot_group, scale_enc, l_enc);
    precom->dec_precom = precompute_s2c(context, ksi_pows, rot_group, scale_dec, l_dec);
}

//==============================================================================
// Precompute linear transform coefficients as encoded plaintexts
//==============================================================================

std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>>
CKKSBootstrapper::precompute_c2s(
    const PhantomContext& context,
    const std::vector<std::complex<double>>& ksi_pows,
    const std::vector<uint32_t>& rot_group,
    double scale, uint32_t level) {

    uint32_t slots = rot_group.size();
    auto precom = precom_map_[slots];

    int32_t level_budget = precom->params_enc[boot_params::LEVEL_BUDGET];
    int32_t layers_coll = precom->params_enc[boot_params::LAYERS_COLL];
    int32_t num_rot = precom->params_enc[boot_params::NUM_ROTATIONS];
    int32_t b = precom->params_enc[boot_params::BABY_STEP];
    int32_t g = precom->params_enc[boot_params::GIANT_STEP];
    int32_t num_rot_rem = precom->params_enc[boot_params::NUM_ROTATIONS_REM];
    int32_t b_rem = precom->params_enc[boot_params::BABY_STEP_REM];
    int32_t g_rem = precom->params_enc[boot_params::GIANT_STEP_REM];
    int32_t rem_coll = precom->params_enc[boot_params::LAYERS_REM];
    int32_t flag_rem = (rem_coll != 0) ? 1 : 0;

    auto coeffs_all = coeff_encoding_collapse(ksi_pows, rot_group, level_budget, false);

    std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>> result(level_budget);

    for (int32_t si = 0; si < level_budget; si++) {
        bool is_rem = (flag_rem && si == 0);
        uint32_t nr = is_rem ? num_rot_rem : num_rot;
        int32_t g_cur = is_rem ? g_rem : g;
        result[si].resize(nr + 1);

        uint32_t encode_level = level - si;
        auto& level_ctx = context.get_context_data(encode_level);
        auto& cm = level_ctx.parms().coeff_modulus();
        double encode_scale = static_cast<double>(cm[cm.size() - 1].value());

        // Compute shift_base matching coeffs_to_slots rotation computation
        int32_t shift_base;
        if (is_rem) {
            shift_base = 1;
        } else {
            shift_base = 1 << ((si - flag_rem) * layers_coll + rem_coll);
        }

        // Apply scale only to the LAST processed layer (matching reference).
        // C2S processes layers from si=levelBudget-1 (first) to si=0 (last).
        // So si=0 (or si=stop for remainder) is the last processed.
        bool apply_scale = (si == 0) || (is_rem);

        for (uint32_t j = 0; j < nr; j++) {
            auto& diag = coeffs_all[si][j];

            int32_t giant_idx = j / g_cur;
            uint32_t rot_out = reduce_rotation(g_cur * giant_idx * shift_base, slots);

            std::vector<cuDoubleComplex> vals(slots);
            for (uint32_t k = 0; k < slots; k++) {
                uint32_t k_src = (k + slots - rot_out) % slots;
                double re = diag[k_src].real();
                double im = diag[k_src].imag();
                if (apply_scale) {
                    re *= scale;
                    im *= scale;
                }
                vals[k] = make_cuDoubleComplex(re, im);
            }

            auto pt = std::make_shared<PhantomPlaintext>();
            encoder_.encode_ext(context, vals, encode_scale, *pt, encode_level);
            result[si][j] = pt;
        }
        result[si][nr] = nullptr;
    }

    return result;
}

std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>>
CKKSBootstrapper::precompute_s2c(
    const PhantomContext& context,
    const std::vector<std::complex<double>>& ksi_pows,
    const std::vector<uint32_t>& rot_group,
    double scale, uint32_t level) {

    uint32_t slots = rot_group.size();
    auto precom = precom_map_[slots];

    int32_t level_budget = precom->params_dec[boot_params::LEVEL_BUDGET];
    int32_t layers_coll = precom->params_dec[boot_params::LAYERS_COLL];
    int32_t num_rot = precom->params_dec[boot_params::NUM_ROTATIONS];
    int32_t b = precom->params_dec[boot_params::BABY_STEP];
    int32_t g = precom->params_dec[boot_params::GIANT_STEP];
    int32_t num_rot_rem = precom->params_dec[boot_params::NUM_ROTATIONS_REM];
    int32_t b_rem = precom->params_dec[boot_params::BABY_STEP_REM];
    int32_t g_rem = precom->params_dec[boot_params::GIANT_STEP_REM];
    int32_t rem_coll = precom->params_dec[boot_params::LAYERS_REM];
    int32_t flag_rem = (rem_coll != 0) ? 1 : 0;

    auto coeffs_all = coeff_decoding_collapse(ksi_pows, rot_group, level_budget, false);

    std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>> result(level_budget);

    for (int32_t si = 0; si < level_budget; si++) {
        bool is_rem = (flag_rem && si == level_budget - 1);
        uint32_t nr = is_rem ? num_rot_rem : num_rot;
        int32_t g_cur = is_rem ? g_rem : g;
        result[si].resize(nr + 1);

        uint32_t encode_level = level + si;
        auto& level_ctx = context.get_context_data(encode_level);
        auto& cm = level_ctx.parms().coeff_modulus();
        double encode_scale = static_cast<double>(cm[cm.size() - 1].value());

        // Compute shift_base matching slots_to_coeffs rotation computation
        int32_t shift_base = 1 << (si * layers_coll);

        // Apply scale only to the LAST processed layer (matching reference).
        // S2C processes layers from si=0 (first) to si=levelBudget-1 (last).
        // For no-remainder: last is si=levelBudget-1.
        // For remainder: last is the remainder layer (si=levelBudget-1).
        bool apply_scale = (si == level_budget - 1);

        for (uint32_t j = 0; j < nr; j++) {
            auto& diag = coeffs_all[si][j];

            int32_t giant_idx = j / g_cur;
            uint32_t rot_out = reduce_rotation(g_cur * giant_idx * shift_base, slots);

            std::vector<cuDoubleComplex> vals(slots);
            for (uint32_t k = 0; k < slots; k++) {
                uint32_t k_src = (k + slots - rot_out) % slots;
                double re = diag[k_src].real();
                double im = diag[k_src].imag();
                if (apply_scale) {
                    re *= scale;
                    im *= scale;
                }
                vals[k] = make_cuDoubleComplex(re, im);
            }

            auto pt = std::make_shared<PhantomPlaintext>();
            encoder_.encode_ext(context, vals, encode_scale, *pt, encode_level);
            result[si][j] = pt;
        }
        result[si][nr] = nullptr;
    }

    return result;
}

//==============================================================================
// Rotation index computation
//==============================================================================

std::vector<int32_t> CKKSBootstrapper::find_rotation_indices(uint32_t slots, uint32_t M) {
    std::vector<int32_t> indices;

    auto precom = precom_map_[slots];
    auto& params_enc = precom->params_enc;
    auto& params_dec = precom->params_dec;

    // encoding rotation indices
    {
        int32_t budget = params_enc[boot_params::LEVEL_BUDGET];
        int32_t layers = params_enc[boot_params::LAYERS_COLL];
        int32_t rem = params_enc[boot_params::LAYERS_REM];
        int32_t num_rot = params_enc[boot_params::NUM_ROTATIONS];
        int32_t b = params_enc[boot_params::BABY_STEP];
        int32_t g = params_enc[boot_params::GIANT_STEP];
        int32_t num_rot_rem = params_enc[boot_params::NUM_ROTATIONS_REM];
        int32_t b_rem = params_enc[boot_params::BABY_STEP_REM];
        int32_t g_rem = params_enc[boot_params::GIANT_STEP_REM];
        int32_t flag_rem = (rem != 0) ? 1 : 0;
        int32_t stop = (rem == 0) ? -1 : 0;

        for (int32_t si = budget - 1; si > stop; si--) {
            for (int32_t j = 0; j < g; j++) {
                int32_t rot = reduce_rotation(
                    (j - int32_t((num_rot + 1) / 2) + 1) *
                    (1 << ((si - flag_rem) * layers + rem)), slots);
                if (rot != 0) indices.push_back(rot);
            }
            for (int32_t i = 1; i < b; i++) {
                int32_t rot = reduce_rotation(
                    (g * i) * (1 << ((si - flag_rem) * layers + rem)), M / 4);
                if (rot != 0) indices.push_back(rot);
            }
        }
        if (flag_rem) {
            for (int32_t j = 0; j < g_rem; j++) {
                int32_t rot = reduce_rotation(
                    (j - int32_t((num_rot_rem + 1) / 2) + 1), slots);
                if (rot != 0) indices.push_back(rot);
            }
            for (int32_t i = 1; i < b_rem; i++) {
                int32_t rot = reduce_rotation(g_rem * i, M / 4);
                if (rot != 0) indices.push_back(rot);
            }
        }
    }

    // decoding rotation indices
    {
        int32_t budget = params_dec[boot_params::LEVEL_BUDGET];
        int32_t layers = params_dec[boot_params::LAYERS_COLL];
        int32_t rem = params_dec[boot_params::LAYERS_REM];
        int32_t num_rot = params_dec[boot_params::NUM_ROTATIONS];
        int32_t b = params_dec[boot_params::BABY_STEP];
        int32_t g = params_dec[boot_params::GIANT_STEP];
        int32_t num_rot_rem = params_dec[boot_params::NUM_ROTATIONS_REM];
        int32_t b_rem = params_dec[boot_params::BABY_STEP_REM];
        int32_t g_rem = params_dec[boot_params::GIANT_STEP_REM];
        int32_t flag_rem = (rem != 0) ? 1 : 0;

        for (int32_t si = 0; si < budget - flag_rem; si++) {
            for (int32_t j = 0; j < g; j++) {
                int32_t rot = reduce_rotation(
                    (j - int32_t((num_rot + 1) / 2) + 1) *
                    (1 << (si * layers)), M / 4);
                if (rot != 0) indices.push_back(rot);
            }
            for (int32_t i = 1; i < b; i++) {
                int32_t rot = reduce_rotation(
                    (g * i) * (1 << (si * layers)), M / 4);
                if (rot != 0) indices.push_back(rot);
            }
        }
        if (flag_rem) {
            int32_t si = budget - flag_rem;
            for (int32_t j = 0; j < g_rem; j++) {
                int32_t rot = reduce_rotation(
                    (j - int32_t((num_rot_rem + 1) / 2) + 1) *
                    (1 << (si * layers)), M / 4);
                if (rot != 0) indices.push_back(rot);
            }
            for (int32_t i = 1; i < b_rem; i++) {
                int32_t rot = reduce_rotation(
                    (g_rem * i) * (1 << (si * layers)), M / 4);
                if (rot != 0) indices.push_back(rot);
            }
        }
    }

    // partial sum rotations for sparse packing
    uint32_t N = M / 2;
    if (slots < N / 2) {
        for (uint32_t j = 1; j < N / (2 * slots); j <<= 1)
            indices.push_back(j * slots);
    }

    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    indices.erase(std::remove(indices.begin(), indices.end(), 0), indices.end());

    return indices;
}

//==============================================================================
// KeyGen
//==============================================================================

void CKKSBootstrapper::keygen(
    const PhantomContext& context,
    PhantomSecretKey& secret_key,
    uint32_t num_slots) {

    relin_key_ = secret_key.gen_relinkey(context);
    galois_key_ = secret_key.create_galois_keys(context);
}

//==============================================================================
// BSGS linear transform using standard rotate/multiply_plain
//==============================================================================

PhantomCiphertext CKKSBootstrapper::bsgs_linear_transform(
    const PhantomContext& context,
    const PhantomCiphertext& ct,
    const std::vector<std::shared_ptr<PhantomPlaintext>>& diag_precom,
    int32_t num_rotations, int32_t b, int32_t g,
    const std::vector<int32_t>& rot_in,
    const std::vector<int32_t>& rot_out) {

    // Baby step: rotate ct by each inner rotation amount
    std::vector<PhantomCiphertext> baby_rotations(g);
    for (int32_t j = 0; j < g; j++) {
        if (rot_in[j] != 0) {
            baby_rotations[j] = rotate(context, ct, rot_in[j], galois_key_);
        } else {
            baby_rotations[j] = ct;
        }
    }

    // Giant step: for each outer step, combine baby steps with coefficients
    PhantomCiphertext result;

    for (int32_t i = 0; i < b; i++) {
        int32_t G = g * i;

        // inner accumulation: sum_j baby_rotations[j] * diag[G+j]
        PhantomCiphertext inner;
        bool first_inner = true;

        for (int32_t j = 0; j < g; j++) {
            if ((G + j) >= int32_t(num_rotations)) continue;
            if (!diag_precom[G + j]) continue;

            PhantomCiphertext term = multiply_plain(context, baby_rotations[j], *diag_precom[G + j]);

            if (first_inner) {
                inner = term;
                first_inner = false;
            } else {
                add_inplace(context, inner, term);
            }
        }

        if (first_inner) continue;

        if (i == 0) {
            result = inner;
        } else {
            if (rot_out[i] != 0) {
                rotate_inplace(context, inner, rot_out[i], galois_key_);
            }
            add_inplace(context, result, inner);
        }
    }

    return result;
}

//==============================================================================
// Hoisted BSGS helper functions
//==============================================================================

// Precompute digit decomposition of c1 for hoisted automorphisms
phantom::util::cuda_auto_ptr<uint64_t> CKKSBootstrapper::fast_rotation_precompute(
    const PhantomContext& context, const PhantomCiphertext& ct) {

    const auto& s = cudaStreamPerThread;
    auto& key_parms = context.get_context_data(0).parms();
    auto n = key_parms.poly_modulus_degree();
    auto scheme = key_parms.scheme();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_parms.coeff_modulus().size();

    auto& rns_tool = context.get_context_data(ct.chain_index()).gpu_rns_tool();
    size_t size_Ql = rns_tool.base_Ql().size();
    size_t size_QlP = size_Ql + size_P;

    auto c1_ptr = ct.data() + ct.coeff_modulus_size() * n;
    auto temp = make_cuda_auto_ptr<uint64_t>(size_Ql * n, s);
    cudaMemcpyAsync(temp.get(), c1_ptr, size_Ql * n * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s);

    // Modup: decompose c1 into digits and extend each to QlP basis
    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();
    auto digits = make_cuda_auto_ptr<uint64_t>(beta * size_QlP * n, s);
    rns_tool.modup(digits.get(), temp.get(), context.gpu_rns_tables(), scheme, s);

    return digits;
}

// Fast rotation in extended QlP basis using pre-computed digits
// Uses Approach A: permute digits BEFORE inner product (required for standard galois keys)
PhantomCiphertext CKKSBootstrapper::fast_rotation_ext(
    const PhantomContext& context, const PhantomCiphertext& ct,
    int32_t rot_step, phantom::util::cuda_auto_ptr<uint64_t>& digits,
    bool add_first) {

    const auto& s = cudaStreamPerThread;
    auto& key_context_data = context.get_context_data(0);
    auto& key_parms = key_context_data.parms();
    auto& key_modulus = key_parms.coeff_modulus();
    size_t N = key_parms.poly_modulus_degree();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_modulus.size();
    size_t size_Q = size_QP - size_P;

    auto& rns_tool = context.get_context_data(ct.chain_index()).gpu_rns_tool();
    size_t size_Ql = rns_tool.base_Ql().size();
    size_t size_QlP = size_Ql + size_P;
    auto modulus_QP = context.gpu_rns_tables().modulus();

    uint32_t autoIndex = FindAutomorphismIndex2nComplex(rot_step, 2 * N);

    auto& key_galois_tool = context.key_galois_tool_;
    auto& galois_elts = key_galois_tool->galois_elts();
    auto iter = find(galois_elts.begin(), galois_elts.end(), autoIndex);
    if (iter == galois_elts.end())
        throw std::logic_error("Galois key not present for rotation step " + std::to_string(rot_step));
    auto elt_index = iter - galois_elts.begin();

    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();
    auto size_QlP_n = size_QlP * N;
    auto size_Ql_n = size_Ql * N;

    auto temp_digits = make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, s);
    for (size_t b = 0; b < beta; b++) {
        key_galois_tool->apply_galois_ntt(
            digits.get() + b * size_QlP_n, size_QlP, elt_index,
            temp_digits.get() + b * size_QlP_n, s);
    }

    auto cx = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, s);
    auto reduction_threshold =
        (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;
    key_switch_inner_prod_c2_and_evk<<<size_QlP_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
        cx.get(), temp_digits.get(),
        galois_key_.get_relin_keys(elt_index).public_keys_ptr(),
        modulus_QP, N, size_QP, size_QP * N, size_QlP, size_QlP_n, size_Q, size_Ql,
        beta, reduction_threshold);

    // If add_first: permute c0 FIRST, then multiply by PModq, then add to cx[0]
    if (add_first) {
        std::vector<arith::Modulus> coeff_mod = key_parms.coeff_modulus();
        std::vector<arith::Modulus> vec_ql(size_Ql);
        std::vector<arith::Modulus> vec_p(size_P);

        for (size_t i = 0; i < size_Ql; i++)
            vec_ql[i] = coeff_mod[i];
        for (size_t i = 0; i < size_P; i++)
            vec_p[i] = coeff_mod[i + size_Q];

        arith::BaseConverter base_convert{arith::RNSBase(vec_ql), arith::RNSBase(vec_p)};
        uint64_t* PModq = base_convert.PModq();

        auto perm_c0 = make_cuda_auto_ptr<uint64_t>(size_Ql_n, s);
        key_galois_tool->apply_galois_ntt(ct.data(), size_Ql, elt_index, perm_c0.get(), s);

        auto temp = make_cuda_auto_ptr<uint64_t>(size_Ql_n, s);
        uint64_t gridDimGlb = N / blockDimGlb.x;

        for (size_t j = 0; j < size_Ql; j++) {
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                perm_c0.get() + j * N, PModq[j], &modulus_QP[j],
                temp.get() + j * N, N, 1);
        }

        // Add σ(c0)*PModq to cx[0] (Ql part only)
        add_to_ct_kernel<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
            cx.get(), temp.get(), rns_tool.base_Ql().base(), N, size_Ql);
    }

    // NO post-permutation needed - digits were already permuted (Approach A)
    PhantomCiphertext result;
    result.resize(2, size_QlP, N, s);
    result.set_chain_index(ct.chain_index());
    result.set_scale(ct.scale());
    result.SetNoiseScaleDeg(ct.GetNoiseScaleDeg());
    result.set_ntt_form(ct.is_ntt_form());

    cudaMemcpyAsync(result.data(), cx.get(), 2 * size_QlP_n * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, s);

    return result;
}

// Extend ciphertext from Ql to QlP basis (multiply Ql by PModq, zero P part)
PhantomCiphertext CKKSBootstrapper::key_switch_ext(
    const PhantomContext& context, const PhantomCiphertext& ct) {

    const auto& s = cudaStreamPerThread;
    auto& key_parms = context.get_context_data(0).parms();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_Q = context.get_context_data(context.get_first_index()).parms().coeff_modulus().size();
    size_t size_QP = key_parms.coeff_modulus().size();

    auto& rns_tool = context.get_context_data(ct.chain_index()).gpu_rns_tool();
    size_t size_Ql = rns_tool.base_Ql().size();
    size_t size_QlP = size_Ql + size_P;
    size_t N = ct.poly_modulus_degree();

    auto modulus_QP = context.gpu_rns_tables().modulus();

    std::vector<arith::Modulus> coeff_mod = key_parms.coeff_modulus();
    std::vector<arith::Modulus> vec_ql(size_Ql);
    std::vector<arith::Modulus> vec_p(size_P);
    for (size_t i = 0; i < size_Ql; i++) vec_ql[i] = coeff_mod[i];
    for (size_t i = 0; i < size_P; i++) vec_p[i] = coeff_mod[i + size_Q];
    arith::BaseConverter base_convert{arith::RNSBase(vec_ql), arith::RNSBase(vec_p)};
    uint64_t* PModq = base_convert.PModq();

    PhantomCiphertext result;
    result.resize(ct.size(), size_QlP, N, s);
    cudaMemsetAsync(result.data(), 0, ct.size() * size_QlP * N * sizeof(uint64_t), s);
    result.set_chain_index(ct.chain_index());
    result.set_scale(ct.scale());
    result.SetNoiseScaleDeg(ct.GetNoiseScaleDeg());
    result.set_ntt_form(ct.is_ntt_form());

    auto rns_coeff_ct = ct.coeff_modulus_size() * N;
    auto rns_coeff_res = size_QlP * N;
    uint64_t gridDimGlb = N / blockDimGlb.x;

    for (size_t i = 0; i < ct.size(); i++) {
        for (size_t j = 0; j < size_Ql; j++) {
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                ct.data() + i * rns_coeff_ct + j * N, PModq[j], &modulus_QP[j],
                result.data() + i * rns_coeff_res + j * N, N, 1);
        }
    }

    return result;
}

// Key switch down: convert from QlP to Ql basis via moddown
PhantomCiphertext CKKSBootstrapper::key_switch_down(
    const PhantomContext& context, const PhantomCiphertext& ct) {

    const auto& s = cudaStreamPerThread;
    auto& key_parms = context.get_context_data(0).parms();
    auto scheme = key_parms.scheme();
    size_t N = key_parms.poly_modulus_degree();

    auto& rns_tool = context.get_context_data(ct.chain_index()).gpu_rns_tool();
    size_t size_Ql = rns_tool.base_Ql().size();
    size_t size_QlP = ct.coeff_modulus_size();

    PhantomCiphertext result;
    result.resize(2, size_Ql, N, s);
    result.set_chain_index(ct.chain_index());
    result.set_scale(ct.scale());
    result.SetNoiseScaleDeg(ct.GetNoiseScaleDeg());
    result.set_ntt_form(ct.is_ntt_form());

    auto size_QlP_n = size_QlP * N;

    for (size_t i = 0; i < 2; i++) {
        rns_tool.moddown_from_NTT(
            result.data() + i * size_Ql * N,
            ct.data() + i * size_QlP_n,
            context.gpu_rns_tables(), scheme, s);
    }

    return result;
}

PhantomCiphertext CKKSBootstrapper::key_switch_down_first(
    const PhantomContext& context, const PhantomCiphertext& ct) {

    const auto& s = cudaStreamPerThread;
    auto& key_parms = context.get_context_data(0).parms();
    auto scheme = key_parms.scheme();
    size_t N = key_parms.poly_modulus_degree();

    auto& rns_tool = context.get_context_data(ct.chain_index()).gpu_rns_tool();
    size_t size_Ql = rns_tool.base_Ql().size();

    PhantomCiphertext result;
    result.resize(1, size_Ql, N, s);
    result.set_chain_index(ct.chain_index());
    result.set_scale(ct.scale());
    result.SetNoiseScaleDeg(ct.GetNoiseScaleDeg());
    result.set_ntt_form(ct.is_ntt_form());

    rns_tool.moddown_from_NTT(
        result.data(), ct.data(),
        context.gpu_rns_tables(), scheme, s);

    return result;
}

// Multiply ciphertext (in QlP) by plaintext (in QlP) in place
void CKKSBootstrapper::mult_ext_inplace(
    const PhantomContext& context, PhantomCiphertext& ct,
    const PhantomPlaintext& pt) {

    const auto& s = cudaStreamPerThread;
    size_t size_Q = context.get_context_data(context.get_first_index()).parms().coeff_modulus().size();
    size_t size_Ql = context.get_context_data(ct.chain_index()).gpu_rns_tool().base_Ql().size();
    size_t size_P = context.get_context_data(0).parms().special_modulus_size();

    auto modulus_QP = context.gpu_rns_tables().modulus();
    auto N = ct.poly_modulus_degree();
    auto coeff_modulus = ct.coeff_modulus_size(); // size_QlP
    auto rns_coeff_count = N * coeff_modulus;

    for (size_t i = 0; i < ct.size(); i++) {
        // Ql part
        uint64_t gridDimGlb = N * size_Ql / blockDimGlb.x;
        multiply_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
            ct.data() + i * rns_coeff_count, pt.data(), modulus_QP,
            ct.data() + i * rns_coeff_count, N, size_Ql);

        // P part
        gridDimGlb = N * size_P / blockDimGlb.x;
        multiply_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
            ct.data() + i * rns_coeff_count + size_Ql * N,
            pt.data() + size_Ql * N,
            modulus_QP + size_Q,
            ct.data() + i * rns_coeff_count + size_Ql * N, N, size_P);
    }

    ct.set_scale(ct.scale() * pt.scale());
}

// Add two ciphertexts in QlP basis
void CKKSBootstrapper::add_ext_inplace(
    const PhantomContext& context, PhantomCiphertext& ct1,
    const PhantomCiphertext& ct2) {

    const auto& s = cudaStreamPerThread;
    size_t size_Q = context.get_context_data(context.get_first_index()).parms().coeff_modulus().size();
    size_t size_Ql = context.get_context_data(ct1.chain_index()).gpu_rns_tool().base_Ql().size();
    size_t size_P = context.get_context_data(0).parms().special_modulus_size();

    auto modulus_QP = context.gpu_rns_tables().modulus();
    auto N = ct1.poly_modulus_degree();
    auto coeff_modulus = ct1.coeff_modulus_size();
    auto rns_coeff_count = N * coeff_modulus;

    for (size_t i = 0; i < ct1.size(); i++) {
        // Ql part
        uint64_t gridDimGlb = N * size_Ql / blockDimGlb.x;
        add_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
            ct1.data() + i * rns_coeff_count,
            ct2.data() + i * rns_coeff_count,
            modulus_QP,
            ct1.data() + i * rns_coeff_count, N, size_Ql);

        // P part
        gridDimGlb = N * size_P / blockDimGlb.x;
        add_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
            ct1.data() + i * rns_coeff_count + size_Ql * N,
            ct2.data() + i * rns_coeff_count + size_Ql * N,
            modulus_QP + size_Q,
            ct1.data() + i * rns_coeff_count + size_Ql * N, N, size_P);
    }
}

//==============================================================================
// Hoisted BSGS linear transform
//==============================================================================

PhantomCiphertext CKKSBootstrapper::bsgs_linear_transform_hoisted(
    const PhantomContext& context,
    const PhantomCiphertext& ct,
    const std::vector<std::shared_ptr<PhantomPlaintext>>& diag_precom,
    int32_t num_rotations, int32_t b, int32_t g,
    const std::vector<int32_t>& rot_in,
    const std::vector<int32_t>& rot_out) {

    const auto& s = cudaStreamPerThread;
    auto& key_parms = context.get_context_data(0).parms();
    size_t N = key_parms.poly_modulus_degree();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_Q = context.get_context_data(context.get_first_index()).parms().coeff_modulus().size();

    auto& rns_tool = context.get_context_data(ct.chain_index()).gpu_rns_tool();
    size_t size_Ql = rns_tool.base_Ql().size();
    size_t size_QlP = size_Ql + size_P;

    auto modulus_QP = context.gpu_rns_tables().modulus();

    auto digits = fast_rotation_precompute(context, ct);
    cudaStreamSynchronize(s);

    // Baby steps
    std::vector<PhantomCiphertext> fast_rotations(g);
    for (int32_t j = 0; j < g; j++) {
        if (rot_in[j] != 0) {
            fast_rotations[j] = fast_rotation_ext(context, ct, rot_in[j], digits, true);
        } else {
            fast_rotations[j] = key_switch_ext(context, ct);
        }
    }

    // Giant steps
    PhantomCiphertext outer;
    PhantomCiphertext first;

    for (int32_t i = 0; i < b; i++) {
        int32_t G = g * i;

        PhantomCiphertext inner;
        bool has_inner = false;

        for (int32_t j = 0; j < g; j++) {
            if ((G + j) >= num_rotations) continue;
            if (!diag_precom[G + j]) continue;

            if (!has_inner) {
                inner = fast_rotations[j];
                mult_ext_inplace(context, inner, *diag_precom[G + j]);
                has_inner = true;
            } else {
                PhantomCiphertext term = fast_rotations[j];
                mult_ext_inplace(context, term, *diag_precom[G + j]);
                add_ext_inplace(context, inner, term);
            }
        }

        if (!has_inner) continue;

        if (i == 0) {
            first = key_switch_down_first(context, inner);
            outer = inner;
            cudaMemsetAsync(outer.data(), 0, size_QlP * N * sizeof(uint64_t), s);
        } else {
            if (rot_out[i] != 0) {
                PhantomCiphertext inner_down = key_switch_down(context, inner);

                uint32_t autoIndex = FindAutomorphismIndex2nComplex(rot_out[i], 2 * N);
                auto d_vec = make_cuda_auto_ptr<uint32_t>(N, s);
                uint64_t gridDimGlb = N / blockDimGlb.x;
                PrecomputeAutoMapKernel<<<gridDimGlb, blockDimGlb, 0, s>>>(N, autoIndex, d_vec.get());

                auto rotated_c0 = make_cuda_auto_ptr<uint64_t>(size_Ql * N, s);
                context.key_galois_tool_->apply_galois_ntt_direct(
                    inner_down.data(), size_Ql, rotated_c0.get(), d_vec.get(), s);

                add_rns_poly<<<(size_Ql * N) / blockDimGlb.x, blockDimGlb, 0, s>>>(
                    first.data(), rotated_c0.get(), modulus_QP,
                    first.data(), N, size_Ql);

                auto inner_digits = fast_rotation_precompute(context, inner_down);
                PhantomCiphertext rotated_ext = fast_rotation_ext(
                    context, inner_down, rot_out[i], inner_digits, false);
                add_ext_inplace(context, outer, rotated_ext);
            } else {
                PhantomCiphertext inner_first = key_switch_down_first(context, inner);
                add_rns_poly<<<(size_Ql * N) / blockDimGlb.x, blockDimGlb, 0, s>>>(
                    first.data(), inner_first.data(), modulus_QP,
                    first.data(), N, size_Ql);

                auto rns_coeff_ext = size_QlP * N;
                add_rns_poly<<<(size_Ql * N) / blockDimGlb.x, blockDimGlb, 0, s>>>(
                    outer.data() + rns_coeff_ext,
                    inner.data() + rns_coeff_ext,
                    modulus_QP,
                    outer.data() + rns_coeff_ext, N, size_Ql);
                add_rns_poly<<<(size_P * N) / blockDimGlb.x, blockDimGlb, 0, s>>>(
                    outer.data() + rns_coeff_ext + size_Ql * N,
                    inner.data() + rns_coeff_ext + size_Ql * N,
                    modulus_QP + size_Q,
                    outer.data() + rns_coeff_ext + size_Ql * N, N, size_P);
            }
        }
    }

    // Key switch down + add first
    PhantomCiphertext result = key_switch_down(context, outer);
    add_rns_poly<<<(size_Ql * N) / blockDimGlb.x, blockDimGlb, 0, s>>>(
        result.data(), first.data(), modulus_QP,
        result.data(), N, size_Ql);

    return result;
}

//==============================================================================
// CoeffsToSlots (homomorphic encoding / iDFT)
//==============================================================================

PhantomCiphertext CKKSBootstrapper::coeffs_to_slots(
    const PhantomContext& context,
    const PhantomCiphertext& ct,
    uint32_t num_slots) {

    auto& parms = context.get_context_data(context.get_first_index()).parms();
    uint32_t N = parms.poly_modulus_degree();
    uint32_t M = 2 * N;
    uint32_t slots = (num_slots == 0) ? N / 2 : num_slots;

    auto precom = precom_map_[slots];
    auto& params = precom->params_enc;

    int32_t level_budget = params[boot_params::LEVEL_BUDGET];
    int32_t layers_coll = params[boot_params::LAYERS_COLL];
    int32_t rem_coll = params[boot_params::LAYERS_REM];
    int32_t num_rot = params[boot_params::NUM_ROTATIONS];
    int32_t b = params[boot_params::BABY_STEP];
    int32_t g = params[boot_params::GIANT_STEP];
    int32_t num_rot_rem = params[boot_params::NUM_ROTATIONS_REM];
    int32_t b_rem = params[boot_params::BABY_STEP_REM];
    int32_t g_rem = params[boot_params::GIANT_STEP_REM];

    int32_t stop = (rem_coll == 0) ? -1 : 0;
    int32_t flag_rem = (rem_coll == 0) ? 0 : 1;

    // compute rotation indices for each level
    std::vector<std::vector<int32_t>> rot_in(level_budget);
    std::vector<std::vector<int32_t>> rot_out(level_budget);

    for (int32_t i = 0; i < level_budget; i++) {
        if (flag_rem && i == 0)
            rot_in[i].resize(num_rot_rem + 1);
        else
            rot_in[i].resize(num_rot + 1);
        rot_out[i].resize(b + b_rem);
    }

    for (int32_t si = level_budget - 1; si > stop; si--) {
        for (int32_t j = 0; j < g; j++)
            rot_in[si][j] = reduce_rotation(
                (j - int32_t((num_rot + 1) / 2) + 1) *
                (1 << ((si - flag_rem) * layers_coll + rem_coll)), slots);
        for (int32_t i = 0; i < b; i++)
            rot_out[si][i] = reduce_rotation(
                (g * i) * (1 << ((si - flag_rem) * layers_coll + rem_coll)), M / 4);
    }
    if (flag_rem) {
        for (int32_t j = 0; j < g_rem; j++)
            rot_in[stop][j] = reduce_rotation(
                (j - int32_t((num_rot_rem + 1) / 2) + 1), slots);
        for (int32_t i = 0; i < b_rem; i++)
            rot_out[stop][i] = reduce_rotation(g_rem * i, M / 4);
    }

    PhantomCiphertext result = ct;

    for (int32_t si = level_budget - 1; si > stop; si--) {
        if (si != level_budget - 1)
            rescale_to_next_inplace(context, result);

        result = bsgs_linear_transform_hoisted(
            context, result, precom->enc_precom[si],
            num_rot, b, g, rot_in[si], rot_out[si]);
    }

    if (flag_rem) {
        rescale_to_next_inplace(context, result);
        result = bsgs_linear_transform_hoisted(
            context, result, precom->enc_precom[stop],
            num_rot_rem, b_rem, g_rem, rot_in[stop], rot_out[stop]);
    }

    return result;
}

//==============================================================================
// SlotsToCoeffs (homomorphic decoding / DFT)
//==============================================================================

PhantomCiphertext CKKSBootstrapper::slots_to_coeffs(
    const PhantomContext& context,
    const PhantomCiphertext& ct,
    uint32_t num_slots) {

    auto& parms = context.get_context_data(context.get_first_index()).parms();
    uint32_t N = parms.poly_modulus_degree();
    uint32_t M = 2 * N;
    uint32_t slots = (num_slots == 0) ? N / 2 : num_slots;

    auto precom = precom_map_[slots];
    auto& params = precom->params_dec;

    int32_t level_budget = params[boot_params::LEVEL_BUDGET];
    int32_t layers_coll = params[boot_params::LAYERS_COLL];
    int32_t rem_coll = params[boot_params::LAYERS_REM];
    int32_t num_rot = params[boot_params::NUM_ROTATIONS];
    int32_t b = params[boot_params::BABY_STEP];
    int32_t g = params[boot_params::GIANT_STEP];
    int32_t num_rot_rem = params[boot_params::NUM_ROTATIONS_REM];
    int32_t b_rem = params[boot_params::BABY_STEP_REM];
    int32_t g_rem = params[boot_params::GIANT_STEP_REM];
    int32_t flag_rem = (rem_coll != 0) ? 1 : 0;

    std::vector<std::vector<int32_t>> rot_in(level_budget);
    std::vector<std::vector<int32_t>> rot_out(level_budget);

    for (int32_t i = 0; i < level_budget; i++) {
        if (flag_rem && i == level_budget - 1)
            rot_in[i].resize(num_rot_rem + 1);
        else
            rot_in[i].resize(num_rot + 1);
        rot_out[i].resize(b + b_rem);
    }

    for (int32_t si = 0; si < level_budget - flag_rem; si++) {
        for (int32_t j = 0; j < g; j++)
            rot_in[si][j] = reduce_rotation(
                (j - int32_t((num_rot + 1) / 2) + 1) *
                (1 << (si * layers_coll)), M / 4);
        for (int32_t i = 0; i < b; i++)
            rot_out[si][i] = reduce_rotation(
                (g * i) * (1 << (si * layers_coll)), M / 4);
    }
    if (flag_rem) {
        int32_t si = level_budget - flag_rem;
        for (int32_t j = 0; j < g_rem; j++)
            rot_in[si][j] = reduce_rotation(
                (j - int32_t((num_rot_rem + 1) / 2) + 1) *
                (1 << (si * layers_coll)), M / 4);
        for (int32_t i = 0; i < b_rem; i++)
            rot_out[si][i] = reduce_rotation(
                (g_rem * i) * (1 << (si * layers_coll)), M / 4);
    }

    PhantomCiphertext result = ct;

    for (int32_t si = 0; si < level_budget - flag_rem; si++) {
        if (si != 0)
            rescale_to_next_inplace(context, result);

        result = bsgs_linear_transform_hoisted(
            context, result, precom->dec_precom[si],
            num_rot, b, g, rot_in[si], rot_out[si]);
    }

    if (flag_rem) {
        rescale_to_next_inplace(context, result);
        int32_t si = level_budget - flag_rem;
        result = bsgs_linear_transform_hoisted(
            context, result, precom->dec_precom[si],
            num_rot_rem, b_rem, g_rem, rot_in[si], rot_out[si]);
    }

    return result;
}

//==============================================================================
// Chebyshev polynomial evaluation (Paterson-Stockmeyer)
//==============================================================================

static uint32_t chebyshev_degree(const std::vector<double>& coeffs) {
    for (int32_t i = coeffs.size() - 1; i >= 0; i--)
        if (std::fabs(coeffs[i]) > 1e-30) return i;
    return 0;
}

PhantomCiphertext CKKSBootstrapper::eval_chebyshev(
    const PhantomContext& context,
    const PhantomCiphertext& ct,
    const std::vector<double>& coeffs,
    double a, double b) {

    uint32_t n = chebyshev_degree(coeffs);
    if (n == 0) return ct;

    // ---- Build Chebyshev polynomials T_1(x) .. T_n(x) using binary tree ----
    // Recurrences:
    //   Even i:  T_i(x) = 2*T_{i/2}(x)^2 - 1
    //   Odd  i:  T_i(x) = 2*T_{floor(i/2)}(x)*T_{ceil(i/2)}(x) - x
    // Depth: ceil(log2(n)) multiplicative levels
    //
    // T[i] stores T_{i+1}(x), so T[0] = T_1(x) = x
    std::vector<PhantomCiphertext> T(n);
    T[0] = ct;  // T_1(x) = x

    for (uint32_t i = 2; i <= n; i++) {
        uint32_t half = i / 2;

        if (i % 2 == 0) {
            // Even: T_i = 2*T_{i/2}^2 - 1
            PhantomCiphertext sq = multiply_and_relin(context, T[half - 1], T[half - 1], relin_key_);
            rescale_to_next_inplace(context, sq);

            T[i - 1] = add(context, sq, sq);  // 2*sq
            add_const(context, T[i - 1], -1.0);  // 2*sq - 1
        } else {
            // Odd: T_i = 2*T_{(i-1)/2}*T_{(i+1)/2} - x
            PhantomCiphertext Tj = T[half - 1];   // T_{(i-1)/2}
            PhantomCiphertext Tj1 = T[half];       // T_{(i+1)/2}

            // Align levels for multiplication
            while (Tj.chain_index() < Tj1.chain_index())
                mod_switch_to_next_inplace(context, Tj);
            while (Tj1.chain_index() < Tj.chain_index())
                mod_switch_to_next_inplace(context, Tj1);
            Tj1.set_scale(Tj.scale());

            PhantomCiphertext prod = multiply_and_relin(context, Tj, Tj1, relin_key_);
            rescale_to_next_inplace(context, prod);

            T[i - 1] = add(context, prod, prod);  // 2*prod

            // Subtract x (align x to the same chain)
            PhantomCiphertext x_copy = ct;
            while (x_copy.chain_index() < T[i - 1].chain_index())
                mod_switch_to_next_inplace(context, x_copy);
            x_copy.set_scale(T[i - 1].scale());
            sub_inplace(context, T[i - 1], x_copy);
        }
    }

    // ---- Linear combination with c0/2 convention ----
    // result = c0/2 + sum_{i=1}^{n} c_i * T_i(x)

    // Multiply each T_i by its coefficient, rescale
    std::vector<PhantomCiphertext> terms;
    terms.reserve(n);
    for (uint32_t i = 1; i <= n; i++) {
        if (std::fabs(coeffs[i]) < 1e-30) continue;

        PhantomCiphertext term = T[i - 1];
        mult_by_const(context, term, coeffs[i]);
        if (term.GetNoiseScaleDeg() >= 2)
            rescale_to_next_inplace(context, term);
        terms.push_back(std::move(term));
    }


    T.clear();
    T.shrink_to_fit();

    // Align all terms to deepest chain index
    size_t max_chain = 0;
    for (auto& t : terms)
        max_chain = std::max(max_chain, t.chain_index());

    for (auto& t : terms) {
        while (t.chain_index() < max_chain)
            mod_switch_to_next_inplace(context, t);
        t.set_scale(terms[0].scale());
    }

    PhantomCiphertext result = terms[0];
    for (size_t i = 1; i < terms.size(); i++)
        add_inplace(context, result, terms[i]);

    // Add constant term with c0/2 convention
    add_const(context, result, coeffs[0] / 2.0);

    return result;
}

//==============================================================================
// Double-angle iterations
//==============================================================================

void CKKSBootstrapper::double_angle_iterations(
    const PhantomContext& context,
    PhantomCiphertext& ct,
    uint32_t num_iter) {

    for (uint32_t j = 1; j <= num_iter; j++) {
        if (ct.GetNoiseScaleDeg() >= 2)
            rescale_to_next_inplace(context, ct);

        PhantomCiphertext ct2 = ct;
        ct2.set_scale(ct.scale());
        PhantomCiphertext sq = multiply_and_relin(context, ct, ct2, relin_key_);
        rescale_to_next_inplace(context, sq);

        ct = add(context, sq, sq); // 2 * ct^2

        double scalar = -1.0 / std::pow(2.0 * M_PI, std::pow(2.0, (int32_t)j - (int32_t)num_iter));
        add_const(context, ct, scalar);
    }
}

//==============================================================================
// Ciphertext adjustment before bootstrap
//==============================================================================

void CKKSBootstrapper::adjust_ciphertext(
    const PhantomContext& context,
    PhantomCiphertext& ct,
    double correction) {

    auto& ctx = context.get_context_data(ct.chain_index());
    auto& parms = ctx.parms();
    uint32_t num_towers = parms.coeff_modulus().size();

    auto& first_parms = context.get_context_data(context.get_first_index()).parms();
    double target_sf = static_cast<double>(first_parms.coeff_modulus()[0].value());
    double source_sf = ct.scale();
    double mod_to_drop = static_cast<double>(
        context.get_context_data(0).parms().coeff_modulus()[num_towers - 1].value());

    double adj_factor = (target_sf / source_sf) * (mod_to_drop / source_sf) * std::pow(2, -correction);
    mult_by_const(context, ct, adj_factor);
    rescale_to_next_inplace(context, ct);
    ct.set_scale(target_sf);
}

//==============================================================================
// Main bootstrap function
//==============================================================================

PhantomCiphertext CKKSBootstrapper::bootstrap(
    const PhantomContext& context,
    PhantomCiphertext& ct,
    uint32_t num_slots) {

    auto& parms = context.get_context_data(context.get_first_index()).parms();
    uint32_t N = parms.poly_modulus_degree();
    uint32_t M = 2 * N;
    uint32_t slots = (num_slots == 0) ? N / 2 : num_slots;

    auto it = precom_map_.find(slots);
    if (it == precom_map_.end())
        throw std::runtime_error("Bootstrap: call setup() first for " + std::to_string(slots) + " slots");

    double q_double = static_cast<double>(parms.coeff_modulus()[0].value());
    double pow_p = std::pow(2, std::round(std::log2(q_double)));
    int32_t deg = std::round(std::log2(q_double / pow_p));
    if (deg < 0) deg = 0;
    if (deg > static_cast<int32_t>(correction_factor_))
        throw std::runtime_error("deg exceeds correction_factor");

    uint32_t correction = correction_factor_ - deg;
    double post = std::pow(2, static_cast<double>(deg));
    double pre = 1.0 / post;
    uint64_t scalar = std::llround(post);

    // ---- MODRAISE ----
    PhantomCiphertext raised = ct;
    if (raised.GetNoiseScaleDeg() > 1)
        rescale_to_next_inplace(context, raised);
    adjust_ciphertext(context, raised, correction);

    // Mod-switch down to exactly 1 remaining prime before modraise.
    // mod_raise only uses q_0 for signed lift; extra primes cause data loss.
    while (raised.coeff_modulus_size() > 1) {
        mod_switch_to_next_inplace(context, raised);
    }

    raised = mod_raise(context, raised);

    // ---- SCALE FOR EVAL MOD ----
    // Two multiplications for sufficient RNS precision with 45-bit primes:
    //   pre/K ≈ 2e-3 → round(2e-3 * 2^45) ≈ 2^35 (35 bits of precision)
    //   1/N ≈ 3e-5 → round(3e-5 * 2^45) ≈ 2^30 (30 bits of precision)
    double scale_mult_1 = pre * (1.0 / K_UNIFORM);
    double scale_mult_2 = 1.0 / N;
    mult_by_const(context, raised, scale_mult_1);
    rescale_to_next_inplace(context, raised);
    mult_by_const(context, raised, scale_mult_2);
    rescale_to_next_inplace(context, raised);

    // ---- COEFFS TO SLOTS ----
    if (slots == M / 4) {
        auto ct_enc = coeffs_to_slots(context, raised, slots);
        rescale_to_next_inplace(context, ct_enc);

        auto conj = conjugate(context, ct_enc);
        conj.set_scale(ct_enc.scale());
        auto ct_enc_i = sub(context, ct_enc, conj);
        add_inplace(context, ct_enc, conj);

        mult_by_monomial(context, ct_enc_i, 3 * M / 4);

        // ---- EVAL MOD (Chebyshev + double angle) ----
        ct_enc = eval_chebyshev(context, ct_enc, chebyshev_coeffs_, -1.0, 1.0);
        ct_enc_i = eval_chebyshev(context, ct_enc_i, chebyshev_coeffs_, -1.0, 1.0);

        double_angle_iterations(context, ct_enc, R_UNIFORM);
        double_angle_iterations(context, ct_enc_i, R_UNIFORM);

        mult_by_monomial(context, ct_enc_i, M / 4);

        while (ct_enc_i.chain_index() < ct_enc.chain_index())
            mod_switch_to_next_inplace(context, ct_enc_i);
        while (ct_enc.chain_index() < ct_enc_i.chain_index())
            mod_switch_to_next_inplace(context, ct_enc);
        ct_enc_i.set_scale(ct_enc.scale());
        add_inplace(context, ct_enc, ct_enc_i);

        mult_by_integer(context, ct_enc, scalar);

        // ---- SLOTS TO COEFFS ----
        auto ct_dec = slots_to_coeffs(context, ct_enc, slots);

        uint64_t cor_factor = (uint64_t)1 << std::llround(correction);
        mult_by_integer(context, ct_dec, cor_factor);

        return ct_dec;
    } else {
        // Sparse case: partial sum first
        for (uint32_t j = 1; j < N / (2 * slots); j <<= 1) {
            auto temp = rotate(context, raised, j * slots, galois_key_);
            raised.set_scale(temp.scale());
            add_inplace(context, raised, temp);
        }

        rescale_to_next_inplace(context, raised);
        auto ct_enc = coeffs_to_slots(context, raised, slots);

        auto conj = conjugate(context, ct_enc);
        ct_enc.set_scale(conj.scale());
        add_inplace(context, ct_enc, conj);

        if (ct_enc.GetNoiseScaleDeg() == 2)
            rescale_to_next_inplace(context, ct_enc);

        ct_enc = eval_chebyshev(context, ct_enc, chebyshev_coeffs_, -1.0, 1.0);

        rescale_to_next_inplace(context, ct_enc);
        double_angle_iterations(context, ct_enc, R_UNIFORM);

        mult_by_integer(context, ct_enc, scalar);
        rescale_to_next_inplace(context, ct_enc);

        auto ct_dec = slots_to_coeffs(context, ct_enc, slots);

        auto rotated = rotate(context, ct_dec, slots, galois_key_);
        ct_dec.set_scale(rotated.scale());
        add_inplace(context, ct_dec, rotated);

        uint64_t cor_factor = (uint64_t)1 << std::llround(correction);
        mult_by_integer(context, ct_dec, cor_factor);

        return ct_dec;
    }
}

PhantomCiphertext CKKSBootstrapper::bootstrap_debug(
    const PhantomContext& context,
    PhantomCiphertext& ct,
    PhantomSecretKey& sk,
    uint32_t num_slots) {

    auto& parms = context.get_context_data(context.get_first_index()).parms();
    uint32_t N = parms.poly_modulus_degree();
    uint32_t M = 2 * N;
    uint32_t slots = (num_slots == 0) ? N / 2 : num_slots;

    auto it = precom_map_.find(slots);
    if (it == precom_map_.end())
        throw std::runtime_error("Bootstrap: call setup() first");

    auto debug_decrypt = [&](const char* label, const PhantomCiphertext& c) {
        PhantomPlaintext pt = sk.decrypt(context, c);
        auto vals = encoder_.decode<cuDoubleComplex>(context, pt);
        std::cout << "[DBG] " << label << " chain=" << c.chain_index()
                  << " scale=" << c.scale() << " nsd=" << c.GetNoiseScaleDeg() << std::endl;
        for (int i = 0; i < 5; i++)
            std::cout << "  [" << i << "] " << vals[i].x << " + " << vals[i].y << "i" << std::endl;
    };

    debug_decrypt("INPUT", ct);

    double q_double = static_cast<double>(parms.coeff_modulus()[0].value());
    double pow_p = std::pow(2, std::round(std::log2(q_double)));
    int32_t deg = std::round(std::log2(q_double / pow_p));
    if (deg < 0) deg = 0;
    uint32_t correction = correction_factor_ - deg;
    double post = std::pow(2, static_cast<double>(deg));
    double pre = 1.0 / post;
    uint64_t scalar = std::llround(post);

    PhantomCiphertext raised = ct;
    if (raised.GetNoiseScaleDeg() > 1)
        rescale_to_next_inplace(context, raised);
    adjust_ciphertext(context, raised, correction);
    debug_decrypt("POST-ADJUST", raised);

    while (raised.coeff_modulus_size() > 1) {
        mod_switch_to_next_inplace(context, raised);
    }

    raised = mod_raise(context, raised);
    debug_decrypt("POST-MODRAISE", raised);

    double scale_mult_1 = pre * (1.0 / K_UNIFORM);
    double scale_mult_2 = 1.0 / N;
    mult_by_const(context, raised, scale_mult_1);
    rescale_to_next_inplace(context, raised);
    debug_decrypt("POST-SCALE-1", raised);
    mult_by_const(context, raised, scale_mult_2);
    rescale_to_next_inplace(context, raised);
    debug_decrypt("POST-SCALE-2", raised);

    auto ct_enc = coeffs_to_slots(context, raised, slots);
    debug_decrypt("POST-C2S", ct_enc);

    rescale_to_next_inplace(context, ct_enc);

    auto conj = conjugate(context, ct_enc);
    conj.set_scale(ct_enc.scale());
    auto ct_enc_i = sub(context, ct_enc, conj);
    add_inplace(context, ct_enc, conj);
    mult_by_monomial(context, ct_enc_i, 3 * M / 4);

    debug_decrypt("REAL", ct_enc);
    debug_decrypt("IMAG", ct_enc_i);

    ct_enc = eval_chebyshev(context, ct_enc, chebyshev_coeffs_, -1.0, 1.0);
    ct_enc_i = eval_chebyshev(context, ct_enc_i, chebyshev_coeffs_, -1.0, 1.0);
    debug_decrypt("POST-CHEBYSHEV REAL", ct_enc);
    debug_decrypt("POST-CHEBYSHEV IMAG", ct_enc_i);

    double_angle_iterations(context, ct_enc, R_UNIFORM);
    double_angle_iterations(context, ct_enc_i, R_UNIFORM);
    debug_decrypt("POST-DOUBLE-ANGLE REAL", ct_enc);
    debug_decrypt("POST-DOUBLE-ANGLE IMAG", ct_enc_i);

    mult_by_monomial(context, ct_enc_i, M / 4);
    while (ct_enc_i.chain_index() < ct_enc.chain_index())
        mod_switch_to_next_inplace(context, ct_enc_i);
    while (ct_enc.chain_index() < ct_enc_i.chain_index())
        mod_switch_to_next_inplace(context, ct_enc);
    ct_enc_i.set_scale(ct_enc.scale());
    add_inplace(context, ct_enc, ct_enc_i);
    debug_decrypt("POST-COMBINE", ct_enc);

    mult_by_integer(context, ct_enc, scalar);
    debug_decrypt("POST-SCALAR", ct_enc);

    auto ct_dec = slots_to_coeffs(context, ct_enc, slots);
    debug_decrypt("POST-S2C", ct_dec);

    uint64_t cor_factor = (uint64_t)1 << std::llround(correction);
    mult_by_integer(context, ct_dec, cor_factor);
    debug_decrypt("FINAL", ct_dec);

    return ct_dec;
}

} // namespace phantom
