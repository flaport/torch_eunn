#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

using namespace torch::indexing;

at::Tensor permute(at::Tensor x);
at::Tensor cmm_forward(at::Tensor x, at::Tensor y);
at::Tensor cmm_backward(at::Tensor x, at::Tensor y);
std::vector<at::Tensor> eunn_forward(at::Tensor angles, at::Tensor x);
std::vector<at::Tensor> eunn_backward(at::Tensor dL_dx, at::Tensor angles, at::Tensor cos_phi, at::Tensor sin_phi, at::Tensor cos_theta, at::Tensor sin_theta, at::Tensor zeros, at::Tensor diag, at::Tensor offdiag, at::Tensor xs);

at::Tensor permute(at::Tensor x) {
    at::Tensor result = at::stack({x.index({Slice(), Slice(1,None,2)}), x.index({Slice(), Slice(0,None,2)})}, 2);
    return at::resize_as_(result, x);
}

at::Tensor cmm_forward(at::Tensor x, at::Tensor y) {
    at::Tensor real = x.index({Ellipsis, 0}) * y.index({Ellipsis, 0}) - x.index({Ellipsis, 1}) * y.index({Ellipsis,1});
    at::Tensor imag = x.index({Ellipsis, 0}) * y.index({Ellipsis, 1}) + x.index({Ellipsis, 1}) * y.index({Ellipsis,0});
    at::Tensor result = at::stack({real, imag}, -1);
    return result;
}

at::Tensor cmm_backward(at::Tensor x, at::Tensor y) {
    at::Tensor real = x.index({Ellipsis, 0}) * y.index({Ellipsis, 0}) + x.index({Ellipsis, 1}) * y.index({Ellipsis,1});
    at::Tensor imag = x.index({Ellipsis, 0}) * y.index({Ellipsis, 1}) - x.index({Ellipsis, 1}) * y.index({Ellipsis,0});
    at::Tensor result = at::stack({real, imag}, -1);
    return result;
}

std::vector<at::Tensor> eunn_forward(at::Tensor angles, at::Tensor x) {
    int b = at::size(x, 0);
    int m = at::size(angles, 0);
    int c = at::size(angles, 1);
    at::Tensor phi = angles.index({Slice(0,None,2)});
    at::Tensor theta = angles.index({Slice(1,None,2)});
    at::Tensor cos_phi = at::cos(phi);
    at::Tensor sin_phi = at::sin(phi);
    at::Tensor cos_theta = at::cos(theta);
    at::Tensor sin_theta = at::sin(theta);
    at::Tensor zeros = at::zeros_like(theta);
    at::Tensor diag_r = at::stack({cos_phi*cos_theta, cos_theta}, 1).view({-1, c});
    at::Tensor diag_i = at::stack({sin_phi*cos_theta, zeros}, 1).view({-1, c});
    at::Tensor diag = at::stack({diag_r, diag_i}, -1).index({None}).permute({2,0,1,3});
    at::Tensor offdiag_r = at::stack({-cos_phi*sin_theta, sin_theta}, 1).view({-1, c});
    at::Tensor offdiag_i = at::stack({-sin_phi*sin_theta, zeros}, 1).view({-1, c});
    at::Tensor offdiag = at::stack({offdiag_r, offdiag_i}, -1).index({None}).permute({2,0,1,3});
    at::Tensor xs = at::zeros_like(x).index({None}).repeat({c,1,1,1});
    for (int i=0; i<c; i++) {
        xs[i] = x;
        x = cmm_forward(diag[i], x) + cmm_forward(offdiag[i], permute(x));
        x = at::roll(x, {2*(i%2)-1}, {1});
    }
    return {x, angles, cos_phi, sin_phi, cos_theta, sin_theta, zeros, diag, offdiag, xs};
}

std::vector<at::Tensor> eunn_backward(at::Tensor dL_dx, at::Tensor angles, at::Tensor cos_phi, at::Tensor sin_phi, at::Tensor cos_theta, at::Tensor sin_theta, at::Tensor zeros, at::Tensor diag, at::Tensor offdiag, at::Tensor xs) {
    at::Tensor x;
    int b = at::size(dL_dx, 0);
    int m = at::size(angles, 0);
    int c = at::size(angles, 1);
    at::Tensor dL_ddiag = at::zeros_like(diag);
    at::Tensor dL_doffdiag = at::zeros_like(offdiag);
    for (int i=c-1; i>=0; i--){
        x = xs[i];
        dL_dx = at::roll(dL_dx, {2 * ((i + 1) % 2) - 1}, {1});
        dL_ddiag[i] = cmm_backward(x, dL_dx).sum({0});
        dL_doffdiag[i] = cmm_backward(permute(x), dL_dx).sum({0});
        dL_dx = cmm_backward(diag[i], dL_dx) + permute(cmm_backward(offdiag[i], dL_dx));
    }
    at::Tensor dL_ddiag1_r = dL_ddiag.index({Slice(), 0, Slice(0,None,2), 0}).t();
    at::Tensor dL_ddiag2_r = dL_ddiag.index({Slice(), 0, Slice(1,None,2), 0}).t();
    at::Tensor dL_ddiag1_i = dL_ddiag.index({Slice(), 0, Slice(0,None,2), 1}).t();
    at::Tensor dL_doffdiag1_r = dL_doffdiag.index({Slice(), 0, Slice(0,None,2), 0}).t();
    at::Tensor dL_doffdiag2_r = dL_doffdiag.index({Slice(), 0, Slice(1,None,2), 0}).t();
    at::Tensor dL_doffdiag1_i = dL_doffdiag.index({Slice(), 0, Slice(0,None,2), 1}).t();

    at::Tensor dL_dphi = -dL_ddiag1_r * sin_phi * cos_theta + dL_ddiag1_i * cos_phi * cos_theta;
    dL_dphi += dL_doffdiag1_r * sin_phi * sin_theta - dL_doffdiag1_i * cos_phi * sin_theta;

    at::Tensor dL_dtheta = -dL_ddiag1_r * cos_phi * sin_theta - dL_ddiag1_i * sin_phi * sin_theta;
    dL_dtheta += -dL_ddiag2_r * sin_theta;
    dL_dtheta += -dL_doffdiag1_r * cos_phi * cos_theta - dL_doffdiag1_i * sin_phi * cos_theta;
    dL_dtheta += dL_doffdiag2_r * cos_theta;

    at::Tensor dL_dangles = at::stack({dL_dphi, dL_dtheta}, 1).view({m,c});
    return {dL_dangles, dL_dx};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("permute",       &permute,       "Pairwise permutation of tensor elements along the feature dimension");
    m.def("cmm_forward",   &cmm_forward,   "Complex elementwise multiplication between two torch tensors");
    m.def("cmm_backward",  &cmm_backward,  "Backward pass for complex elementwise multiplication between two torch tensors");
    m.def("eunn_forward",  &eunn_forward,  "Perform the action of a unitary matrix U on x");
    m.def("eunn_backward", &eunn_backward, "Perform the backward action of a unitary matrix U on x");
}
