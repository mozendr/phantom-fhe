#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <complex>

#include "phantom.h"
#include "bootstrap.cuh"

namespace py = pybind11;

namespace pybind11 { namespace detail {
    template <> struct type_caster<cuDoubleComplex> {
    public:
        PYBIND11_TYPE_CASTER(cuDoubleComplex, const_name("complex"));

        bool load(handle src, bool) {
            if (!src) return false;
            if (PyComplex_Check(src.ptr())) {
                value.x = PyComplex_RealAsDouble(src.ptr());
                value.y = PyComplex_ImagAsDouble(src.ptr());
                return true;
            }
            if (PyFloat_Check(src.ptr()) || PyLong_Check(src.ptr())) {
                value.x = PyFloat_AsDouble(src.ptr());
                value.y = 0.0;
                return true;
            }
            if (PyTuple_Check(src.ptr()) || PyList_Check(src.ptr())) {
                py::sequence seq = py::reinterpret_borrow<py::sequence>(src);
                if (seq.size() == 2) {
                    value.x = seq[0].cast<double>();
                    value.y = seq[1].cast<double>();
                    return true;
                }
            }
            return false;
        }

        static handle cast(cuDoubleComplex src, return_value_policy, handle) {
            return PyComplex_FromDoubles(src.x, src.y);
        }
    };
}}

PYBIND11_MODULE(pyPhantom, m) {

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::enum_<phantom::scheme_type>(m, "scheme_type")
            .value("none", phantom::scheme_type::none)
            .value("bgv", phantom::scheme_type::bgv)
            .value("bfv", phantom::scheme_type::bfv)
            .value("ckks", phantom::scheme_type::ckks)
            .export_values();

    py::enum_<phantom::mul_tech_type>(m, "mul_tech_type")
            .value("none", phantom::mul_tech_type::none)
            .value("behz", phantom::mul_tech_type::behz)
            .value("hps", phantom::mul_tech_type::hps)
            .value("hps_overq", phantom::mul_tech_type::hps_overq)
            .value("hps_overq_leveled", phantom::mul_tech_type::hps_overq_leveled)
            .export_values();

    py::enum_<phantom::arith::sec_level_type>(m, "sec_level_type")
            .value("none", phantom::arith::sec_level_type::none)
            .value("tc128", phantom::arith::sec_level_type::tc128)
            .value("tc192", phantom::arith::sec_level_type::tc192)
            .value("tc256", phantom::arith::sec_level_type::tc256)
            .export_values();

    py::class_<phantom::arith::Modulus>(m, "modulus")
            .def(py::init<std::uint64_t>());

    m.def("create_coeff_modulus", &phantom::arith::CoeffModulus::Create);

    m.def("create_plain_modulus", &phantom::arith::PlainModulus::Batching);

    py::class_<phantom::EncryptionParameters>(m, "params")
            .def(py::init<phantom::scheme_type>())
            .def("set_mul_tech", &phantom::EncryptionParameters::set_mul_tech)
            .def("set_poly_modulus_degree", &phantom::EncryptionParameters::set_poly_modulus_degree)
            .def("set_special_modulus_size", &phantom::EncryptionParameters::set_special_modulus_size)
            .def("set_galois_elts", &phantom::EncryptionParameters::set_galois_elts)
            .def("set_coeff_modulus", &phantom::EncryptionParameters::set_coeff_modulus)
            .def("set_plain_modulus", &phantom::EncryptionParameters::set_plain_modulus);

    py::class_<phantom::util::cuda_stream_wrapper>(m, "cuda_stream")
            .def(py::init<>());

    py::class_<PhantomContext>(m, "context")
            .def(py::init<phantom::EncryptionParameters &>());

    py::class_<PhantomSecretKey>(m, "secret_key")
            .def(py::init<const PhantomContext &>())
            .def("gen_publickey", &PhantomSecretKey::gen_publickey)
            .def("gen_relinkey", &PhantomSecretKey::gen_relinkey)
            .def("create_galois_keys", &PhantomSecretKey::create_galois_keys)
            .def("encrypt_symmetric",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomSecretKey::encrypt_symmetric, py::const_), py::arg(), py::arg())
            .def("decrypt",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &>(
                         &PhantomSecretKey::decrypt), py::arg(), py::arg());

    py::class_<PhantomPublicKey>(m, "public_key")
            .def(py::init<>())
            .def("encrypt_asymmetric",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomPublicKey::encrypt_asymmetric), py::arg(), py::arg());

    py::class_<PhantomRelinKey>(m, "relin_key")
            .def(py::init<>());

    py::class_<PhantomGaloisKey>(m, "galois_key")
            .def(py::init<>());

    m.def("get_elt_from_step", &phantom::util::get_elt_from_step);

    m.def("get_elts_from_steps", &phantom::util::get_elts_from_steps);

    py::class_<PhantomBatchEncoder>(m, "batch_encoder")
            .def(py::init<const PhantomContext &>())
            .def("slot_count", &PhantomBatchEncoder::slot_count)
            .def("encode",
                 py::overload_cast<const PhantomContext &, const std::vector<uint64_t> &>(
                         &PhantomBatchEncoder::encode, py::const_), py::arg(), py::arg())
            .def("decode",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomBatchEncoder::decode, py::const_), py::arg(), py::arg());

    py::class_<PhantomCKKSEncoder>(m, "ckks_encoder")
            .def(py::init<const PhantomContext &>())
            .def("slot_count", &PhantomCKKSEncoder::slot_count)
            .def("encode_complex_vector",
                 py::overload_cast<const PhantomContext &, const std::vector<cuDoubleComplex> &, double, size_t>(
                         &PhantomCKKSEncoder::encode<cuDoubleComplex>),
                 py::arg(), py::arg(), py::arg(), py::arg("chain_index") = 1)
            .def("encode_double_vector",
                 py::overload_cast<const PhantomContext &, const std::vector<double> &, double, size_t>(
                         &PhantomCKKSEncoder::encode<double>),
                 py::arg(), py::arg(), py::arg(),
                 py::arg("chain_index") = 1)
            .def("decode_complex_vector",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomCKKSEncoder::decode<cuDoubleComplex>),
                 py::arg(), py::arg())
            .def("decode_double_vector",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomCKKSEncoder::decode<double>), py::arg(), py::arg())
            .def("encode_double_vector_batch",
                 [](PhantomCKKSEncoder &self, const PhantomContext &ctx,
                    py::array_t<double, py::array::c_style | py::array::forcecast> batch_data,
                    double scale, size_t chain_index) {
                     auto buf = batch_data.request();
                     if (buf.ndim != 2)
                         throw std::invalid_argument("batch_data must be a 2D numpy array");
                     return self.encode_batch(ctx,
                                              static_cast<const double*>(buf.ptr),
                                              buf.shape[0], buf.shape[1],
                                              scale, chain_index);
                 },
                 py::arg("ctx"), py::arg("batch_data"), py::arg("scale"),
                 py::arg("chain_index") = 1)
            .def("encode_complex_vector_batch",
                 [](PhantomCKKSEncoder &self, const PhantomContext &ctx,
                    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> batch_data,
                    double scale, size_t chain_index) {
                     auto buf = batch_data.request();
                     if (buf.ndim != 2)
                         throw std::invalid_argument("batch_data must be a 2D numpy array");
                     return self.encode_batch_complex(ctx,
                                                      reinterpret_cast<const cuDoubleComplex*>(buf.ptr),
                                                      buf.shape[0], buf.shape[1],
                                                      scale, chain_index);
                 },
                 py::arg("ctx"), py::arg("batch_data"), py::arg("scale"),
                 py::arg("chain_index") = 1);

    py::class_<PhantomPlaintext>(m, "plaintext")
            .def(py::init<>())
            .def("chain_index", &PhantomPlaintext::chain_index)
            .def("scale", &PhantomPlaintext::scale)
            .def("coeff_modulus_size", &PhantomPlaintext::coeff_modulus_size)
            .def("poly_modulus_degree", &PhantomPlaintext::poly_modulus_degree)
            .def("set_chain_index", &PhantomPlaintext::set_chain_index)
            .def("set_scale", &PhantomPlaintext::set_scale);

    m.def("offload_plaintexts",
        [](py::list pts) -> py::tuple {
            size_t n = pts.size();
            if (n == 0) throw std::invalid_argument("empty plaintext list");

            auto& first = pts[0].cast<PhantomPlaintext&>();
            size_t cms = first.coeff_modulus_size();
            size_t pmd = first.poly_modulus_degree();
            size_t chain_idx = first.chain_index();
            double scale = first.scale();
            size_t elem_count = cms * pmd;
            size_t total_bytes = n * elem_count * sizeof(uint64_t);

            uint64_t* d_staging;
            cudaMalloc(&d_staging, total_bytes);
            for (size_t i = 0; i < n; i++) {
                auto& pt = pts[i].cast<PhantomPlaintext&>();
                cudaMemcpyAsync(d_staging + i * elem_count, pt.data(),
                                elem_count * sizeof(uint64_t),
                                cudaMemcpyDeviceToDevice, cudaStreamPerThread);
            }

            auto result = py::array_t<uint64_t>({(py::ssize_t)n, (py::ssize_t)elem_count});
            auto buf = result.request();
            uint64_t* h_pinned;
            cudaMallocHost(&h_pinned, total_bytes);
            cudaStreamSynchronize(cudaStreamPerThread);
            cudaMemcpy(h_pinned, d_staging, total_bytes, cudaMemcpyDeviceToHost);
            memcpy(buf.ptr, h_pinned, total_bytes);

            cudaFreeHost(h_pinned);
            cudaFree(d_staging);
            return py::make_tuple(result, chain_idx, scale, cms, pmd);
        },
        py::arg("pts"));

    m.def("upload_plaintexts",
        [](py::array_t<uint64_t, py::array::c_style | py::array::forcecast> data,
           size_t chain_index, double scale,
           size_t cms, size_t pmd) -> py::list {
            auto buf = data.request();
            if (buf.ndim != 2)
                throw std::invalid_argument("data must be a 2D numpy array");

            size_t n = buf.shape[0];
            size_t elem_count = buf.shape[1];
            if (elem_count != cms * pmd)
                throw std::invalid_argument("element count mismatch: expected cms*pmd");

            size_t total_bytes = n * elem_count * sizeof(uint64_t);

            uint64_t* h_pinned;
            cudaMallocHost(&h_pinned, total_bytes);
            memcpy(h_pinned, buf.ptr, total_bytes);

            uint64_t* d_staging;
            cudaMalloc(&d_staging, total_bytes);
            cudaMemcpy(d_staging, h_pinned, total_bytes, cudaMemcpyHostToDevice);
            cudaFreeHost(h_pinned);

            std::vector<PhantomPlaintext> pts_vec(n);
            for (size_t i = 0; i < n; i++) {
                pts_vec[i].resize(cms, pmd, cudaStreamPerThread);
            }

            for (size_t i = 0; i < n; i++) {
                cudaMemcpyAsync(pts_vec[i].data(), d_staging + i * elem_count,
                                elem_count * sizeof(uint64_t),
                                cudaMemcpyDeviceToDevice, cudaStreamPerThread);
                pts_vec[i].set_chain_index(chain_index);
                pts_vec[i].set_scale(scale);
            }

            cudaStreamSynchronize(cudaStreamPerThread);
            cudaFree(d_staging);

            py::list result;
            for (size_t i = 0; i < n; i++) {
                result.append(std::move(pts_vec[i]));
            }
            return result;
        },
        py::arg("data"), py::arg("chain_index"), py::arg("scale"),
        py::arg("cms"), py::arg("pmd"));

    py::class_<PhantomCiphertext>(m, "ciphertext")
            .def(py::init<>())
            .def("set_scale", &PhantomCiphertext::set_scale)
            .def("chain_index", &PhantomCiphertext::chain_index)
            .def("coeff_modulus_size", &PhantomCiphertext::coeff_modulus_size)
            .def("scale", &PhantomCiphertext::scale);

    m.def("negate", &phantom::negate, py::arg(), py::arg());

    m.def("add", &phantom::add, py::arg(), py::arg(), py::arg());

    m.def("add_plain", &phantom::add_plain, py::arg(), py::arg(), py::arg());

    m.def("add_many", &phantom::add_many, py::arg(), py::arg(), py::arg());

    m.def("sub", &phantom::sub, py::arg(), py::arg(), py::arg(), py::arg("negate") = false);

    m.def("sub_plain", &phantom::sub_plain, py::arg(), py::arg(), py::arg());

    m.def("multiply", &phantom::multiply, py::arg(), py::arg(), py::arg());

    m.def("multiply_and_relin", &phantom::multiply_and_relin, py::arg(), py::arg(), py::arg(), py::arg());

    m.def("multiply_plain", &phantom::multiply_plain, py::arg(), py::arg(), py::arg());

    m.def("relinearize", &phantom::relinearize, py::arg(), py::arg(), py::arg());

    m.def("rescale_to_next", &phantom::rescale_to_next, py::arg(), py::arg());

    m.def("mod_switch_to_next",
          py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(&phantom::mod_switch_to_next),
          py::arg(), py::arg());

    m.def("mod_switch_to_next",
          py::overload_cast<const PhantomContext &, const PhantomCiphertext &>(&phantom::mod_switch_to_next),
          py::arg(), py::arg());

    m.def("mod_switch_to", py::overload_cast<const PhantomContext &, const PhantomPlaintext &, size_t>(
            &phantom::mod_switch_to), py::arg(), py::arg(), py::arg());

    m.def("mod_switch_to", py::overload_cast<const PhantomContext &, const PhantomCiphertext &, size_t>(
            &phantom::mod_switch_to), py::arg(), py::arg(), py::arg());

    m.def("apply_galois", &phantom::apply_galois, py::arg(), py::arg(), py::arg(), py::arg());

    m.def("rotate", &phantom::rotate, py::arg(), py::arg(), py::arg(), py::arg());

    m.def("hoisting", &phantom::hoisting, py::arg(), py::arg(), py::arg(), py::arg());

    py::class_<phantom::CKKSBootstrapper>(m, "ckks_bootstrapper")
            .def(py::init<PhantomCKKSEncoder &>())
            .def("setup", &phantom::CKKSBootstrapper::setup,
                 py::arg(), py::arg(),
                 py::arg("dim1") = std::vector<uint32_t>{0, 0},
                 py::arg("num_slots") = 0,
                 py::arg("correction_factor") = 0)
            .def("keygen", &phantom::CKKSBootstrapper::keygen,
                 py::arg(), py::arg(), py::arg("num_slots") = 0)
            .def("bootstrap", &phantom::CKKSBootstrapper::bootstrap,
                 py::arg(), py::arg(), py::arg("num_slots") = 0)
            .def("coeffs_to_slots", &phantom::CKKSBootstrapper::coeffs_to_slots,
                 py::arg(), py::arg(), py::arg("num_slots") = 0)
            .def("slots_to_coeffs", &phantom::CKKSBootstrapper::slots_to_coeffs,
                 py::arg(), py::arg(), py::arg("num_slots") = 0)
            .def_static("get_bootstrap_depth", &phantom::CKKSBootstrapper::get_bootstrap_depth)
            .def_static("get_galois_elements", &phantom::CKKSBootstrapper::get_galois_elements);

    // BSGS inner loop in C++: multiply_plain + add + giant rotations + rescale
    m.def("bsgs_multiply_accumulate",
        [](const PhantomContext &ctx,
           py::list ct_babies_list,
           py::list diags_list,
           size_t G, size_t B, size_t D,
           const PhantomGaloisKey &gk) -> PhantomCiphertext {

            if (ct_babies_list.size() < static_cast<py::ssize_t>(G))
                throw std::invalid_argument("ct_babies must have at least G elements");
            if (diags_list.size() < static_cast<py::ssize_t>(D))
                throw std::invalid_argument("diags must have at least D elements");

            std::vector<const PhantomCiphertext*> ct_babies(G);
            std::vector<const PhantomPlaintext*> diags(D);

            for (size_t b = 0; b < G; b++)
                ct_babies[b] = &ct_babies_list[b].cast<const PhantomCiphertext&>();
            for (size_t k = 0; k < D; k++)
                diags[k] = &diags_list[k].cast<const PhantomPlaintext&>();

            PhantomCiphertext result;
            bool result_init = false;

            {
                py::gil_scoped_release release;

                for (size_t g = 0; g < B; g++) {
                    PhantomCiphertext inner;
                    bool inner_init = false;

                    for (size_t b = 0; b < G; b++) {
                        size_t k = g * G + b;
                        if (k >= D) continue;

                        auto term = phantom::multiply_plain(ctx, *ct_babies[b], *diags[k]);

                        if (!inner_init) {
                            inner = std::move(term);
                            inner_init = true;
                        } else {
                            phantom::add_inplace(ctx, inner, term);
                        }
                    }

                    if (!inner_init) continue;

                    if (g > 0) {
                        phantom::rotate_inplace(ctx, inner,
                                                static_cast<int>(g * G), gk);
                    }

                    if (!result_init) {
                        result = std::move(inner);
                        result_init = true;
                    } else {
                        phantom::add_inplace(ctx, result, inner);
                    }
                }

                phantom::rescale_to_next_inplace(ctx, result);
            }

            return result;
        },
        py::arg("ctx"), py::arg("ct_babies"), py::arg("diags"),
        py::arg("G"), py::arg("B"), py::arg("D"), py::arg("gk"));

    // BSGS from CPU-offloaded diagonals with persistent staging buffer
    m.def("bsgs_from_cpu",
        [](const PhantomContext &ctx,
           py::list ct_babies_list,
           py::array_t<uint64_t, py::array::c_style | py::array::forcecast> cpu_diags,
           size_t chain_index, double scale,
           size_t cms, size_t pmd,
           size_t G, size_t B, size_t D,
           const PhantomGaloisKey &gk) -> PhantomCiphertext {

            if (ct_babies_list.size() < static_cast<py::ssize_t>(G))
                throw std::invalid_argument("ct_babies must have at least G elements");

            auto buf = cpu_diags.request();
            if (buf.ndim != 2 || static_cast<size_t>(buf.shape[0]) < D)
                throw std::invalid_argument("cpu_diags must be (D, elem_count) uint64 array");

            size_t elem_count = cms * pmd;
            size_t total_bytes = D * elem_count * sizeof(uint64_t);

            std::vector<const PhantomCiphertext*> ct_babies(G);
            for (size_t b = 0; b < G; b++)
                ct_babies[b] = &ct_babies_list[b].cast<const PhantomCiphertext&>();

            PhantomCiphertext result;
            bool result_init = false;

            {
                py::gil_scoped_release release;

                // Persistent thread-local staging buffer
                thread_local uint64_t* d_staging = nullptr;
                thread_local size_t staging_capacity = 0;
                if (total_bytes > staging_capacity) {
                    if (d_staging) cudaFree(d_staging);
                    cudaMalloc(&d_staging, total_bytes);
                    staging_capacity = total_bytes;
                }

                // H2D
                cudaMemcpy(d_staging, buf.ptr, total_bytes, cudaMemcpyHostToDevice);

                // BSGS inner loop
                for (size_t g = 0; g < B; g++) {
                    PhantomCiphertext inner;
                    bool inner_init = false;

                    for (size_t b = 0; b < G; b++) {
                        size_t k = g * G + b;
                        if (k >= D) continue;

                        PhantomPlaintext pt;
                        pt.set_data_view(d_staging + k * elem_count,
                                         cms, pmd, cudaStreamPerThread);
                        pt.set_chain_index(chain_index);
                        pt.set_scale(scale);

                        auto term = phantom::multiply_plain(ctx, *ct_babies[b], pt);
                        pt.release_data_view();

                        if (!inner_init) {
                            inner = std::move(term);
                            inner_init = true;
                        } else {
                            phantom::add_inplace(ctx, inner, term);
                        }
                    }

                    if (!inner_init) continue;

                    if (g > 0) {
                        phantom::rotate_inplace(ctx, inner,
                                                static_cast<int>(g * G), gk);
                    }

                    if (!result_init) {
                        result = std::move(inner);
                        result_init = true;
                    } else {
                        phantom::add_inplace(ctx, result, inner);
                    }
                }

                phantom::rescale_to_next_inplace(ctx, result);
            }

            return result;
        },
        py::arg("ctx"), py::arg("ct_babies"), py::arg("cpu_diags"),
        py::arg("chain_index"), py::arg("scale"),
        py::arg("cms"), py::arg("pmd"),
        py::arg("G"), py::arg("B"), py::arg("D"), py::arg("gk"));

    // Complete BSGS: baby rotations + H2D + inner loop + rescale, all GIL-released
    m.def("bsgs_complete_from_cpu",
        [](const PhantomContext &ctx,
           const PhantomCiphertext &ct_x,
           py::array_t<uint64_t, py::array::c_style | py::array::forcecast> cpu_diags,
           size_t chain_index, double scale,
           size_t cms, size_t pmd,
           size_t G, size_t B, size_t D,
           const PhantomGaloisKey &gk) -> PhantomCiphertext {

            auto buf = cpu_diags.request();
            if (buf.ndim != 2 || static_cast<size_t>(buf.shape[0]) < D)
                throw std::invalid_argument("cpu_diags must be (D, elem_count) uint64 array");

            size_t elem_count = cms * pmd;
            size_t total_bytes = D * elem_count * sizeof(uint64_t);

            PhantomCiphertext result;
            bool result_init = false;

            {
                py::gil_scoped_release release;

                // Baby rotations
                std::vector<PhantomCiphertext> ct_babies(G);
                ct_babies[0] = ct_x;
                for (size_t b = 1; b < G; b++) {
                    ct_babies[b] = phantom::rotate(ctx, ct_x,
                                                    static_cast<int>(b), gk);
                }

                // H2D
                uint64_t* d_staging;
                cudaMalloc(&d_staging, total_bytes);
                cudaMemcpy(d_staging, buf.ptr, total_bytes,
                           cudaMemcpyHostToDevice);

                // BSGS inner loop
                for (size_t g = 0; g < B; g++) {
                    PhantomCiphertext inner;
                    bool inner_init = false;

                    for (size_t b = 0; b < G; b++) {
                        size_t k = g * G + b;
                        if (k >= D) continue;

                        PhantomPlaintext pt;
                        pt.set_data_view(d_staging + k * elem_count,
                                         cms, pmd, cudaStreamPerThread);
                        pt.set_chain_index(chain_index);
                        pt.set_scale(scale);

                        auto term = phantom::multiply_plain(ctx,
                                                             ct_babies[b], pt);
                        pt.release_data_view();

                        if (!inner_init) {
                            inner = std::move(term);
                            inner_init = true;
                        } else {
                            phantom::add_inplace(ctx, inner, term);
                        }
                    }

                    if (!inner_init) continue;

                    if (g > 0) {
                        phantom::rotate_inplace(ctx, inner,
                                                static_cast<int>(g * G), gk);
                    }

                    if (!result_init) {
                        result = std::move(inner);
                        result_init = true;
                    } else {
                        phantom::add_inplace(ctx, result, inner);
                    }
                }

                phantom::rescale_to_next_inplace(ctx, result);
                cudaFree(d_staging);
            }

            return result;
        },
        py::arg("ctx"), py::arg("ct_x"), py::arg("cpu_diags"),
        py::arg("chain_index"), py::arg("scale"),
        py::arg("cms"), py::arg("pmd"),
        py::arg("G"), py::arg("B"), py::arg("D"), py::arg("gk"));
}
