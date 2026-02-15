import ctypes
import os
from typing import Optional, Sequence, Tuple


class TensorLib:
    def __init__(self, path: Optional[str] = None):
        if path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            # Prefer local macOS dylib; fall back to the provided .so artifact name.
            local_dylib = os.path.join(here, "libtensor.dylib")
            local_so = os.path.join(here, "libtensor.so")
            path = local_dylib if os.path.exists(local_dylib) else local_so
        self.path = path
        mode = getattr(ctypes, "RTLD_GLOBAL", None)
        if mode is None:
            self.lib = ctypes.CDLL(self.path)
        else:
            self.lib = ctypes.CDLL(self.path, mode=mode)

        self.tensor_handle = ctypes.c_void_p

        self.lib.tensor_last_error.restype = ctypes.c_char_p

        self.lib.tensor_create.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int,
        ]
        self.lib.tensor_create.restype = self.tensor_handle

        self.lib.tensor_free.argtypes = [self.tensor_handle]
        self.lib.tensor_free.restype = None

        self.lib.tensor_ndim.argtypes = [self.tensor_handle]
        self.lib.tensor_ndim.restype = ctypes.c_int64

        self.lib.tensor_size.argtypes = [self.tensor_handle]
        self.lib.tensor_size.restype = ctypes.c_int64

        self.lib.tensor_shape.argtypes = [self.tensor_handle, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
        self.lib.tensor_shape.restype = ctypes.c_int

        self.lib.tensor_read_data.argtypes = [self.tensor_handle, ctypes.POINTER(ctypes.c_double), ctypes.c_int64]
        self.lib.tensor_read_data.restype = ctypes.c_int

        # Optional newer APIs (present if libtensor was rebuilt)
        if hasattr(self.lib, "tensor_has_grad"):
            self.lib.tensor_has_grad.argtypes = [self.tensor_handle]
            self.lib.tensor_has_grad.restype = ctypes.c_int

        if hasattr(self.lib, "tensor_write_data"):
            self.lib.tensor_write_data.argtypes = [self.tensor_handle, ctypes.POINTER(ctypes.c_double), ctypes.c_int64]
            self.lib.tensor_write_data.restype = ctypes.c_int

        self.lib.tensor_read_grad.argtypes = [self.tensor_handle, ctypes.POINTER(ctypes.c_double), ctypes.c_int64]
        self.lib.tensor_read_grad.restype = ctypes.c_int

        self.lib.tensor_zero_grad.argtypes = [self.tensor_handle]
        self.lib.tensor_zero_grad.restype = None

        self.lib.tensor_backward.argtypes = [self.tensor_handle]
        self.lib.tensor_backward.restype = ctypes.c_int

        self.lib.tensor_backward_scalar.argtypes = [self.tensor_handle, ctypes.c_double]
        self.lib.tensor_backward_scalar.restype = ctypes.c_int

        self.lib.tensor_backward_with_tensor.argtypes = [self.tensor_handle, self.tensor_handle]
        self.lib.tensor_backward_with_tensor.restype = ctypes.c_int

        self.lib.tensor_detach.argtypes = [self.tensor_handle]
        self.lib.tensor_detach.restype = self.tensor_handle

        self.lib.tensor_zeros.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int]
        self.lib.tensor_zeros.restype = self.tensor_handle

        self.lib.tensor_ones.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int]
        self.lib.tensor_ones.restype = self.tensor_handle

        self.lib.tensor_randn.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.lib.tensor_randn.restype = self.tensor_handle

        self.lib.tensor_reshape.argtypes = [self.tensor_handle, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
        self.lib.tensor_reshape.restype = self.tensor_handle

        self.lib.tensor_transpose.argtypes = [self.tensor_handle]
        self.lib.tensor_transpose.restype = self.tensor_handle

        self.lib.tensor_sum.argtypes = [self.tensor_handle]
        self.lib.tensor_sum.restype = self.tensor_handle

        self.lib.tensor_neg.argtypes = [self.tensor_handle]
        self.lib.tensor_neg.restype = self.tensor_handle

        self.lib.tensor_add.argtypes = [self.tensor_handle, self.tensor_handle]
        self.lib.tensor_add.restype = self.tensor_handle

        self.lib.tensor_add_scalar.argtypes = [self.tensor_handle, ctypes.c_double]
        self.lib.tensor_add_scalar.restype = self.tensor_handle

        self.lib.tensor_sub.argtypes = [self.tensor_handle, self.tensor_handle]
        self.lib.tensor_sub.restype = self.tensor_handle

        self.lib.tensor_sub_scalar.argtypes = [self.tensor_handle, ctypes.c_double]
        self.lib.tensor_sub_scalar.restype = self.tensor_handle

        self.lib.tensor_mul.argtypes = [self.tensor_handle, self.tensor_handle]
        self.lib.tensor_mul.restype = self.tensor_handle

        self.lib.tensor_mul_scalar.argtypes = [self.tensor_handle, ctypes.c_double]
        self.lib.tensor_mul_scalar.restype = self.tensor_handle

        self.lib.tensor_div.argtypes = [self.tensor_handle, self.tensor_handle]
        self.lib.tensor_div.restype = self.tensor_handle

        self.lib.tensor_div_scalar.argtypes = [self.tensor_handle, ctypes.c_double]
        self.lib.tensor_div_scalar.restype = self.tensor_handle

        self.lib.tensor_pow.argtypes = [self.tensor_handle, ctypes.c_double]
        self.lib.tensor_pow.restype = self.tensor_handle

        self.lib.tensor_matmul.argtypes = [self.tensor_handle, self.tensor_handle]
        self.lib.tensor_matmul.restype = self.tensor_handle

        self.lib.tensor_relu.argtypes = [self.tensor_handle]
        self.lib.tensor_relu.restype = self.tensor_handle

        # Added ops (present if libtensor was rebuilt)
        if hasattr(self.lib, "tensor_dropout"):
            self.lib.tensor_dropout.argtypes = [
                self.tensor_handle,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int,
            ]
            self.lib.tensor_dropout.restype = self.tensor_handle

        if hasattr(self.lib, "tensor_batchnorm1d"):
            self.lib.tensor_batchnorm1d.argtypes = [
                self.tensor_handle,
                self.tensor_handle,
                self.tensor_handle,
                self.tensor_handle,
                self.tensor_handle,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
            ]
            self.lib.tensor_batchnorm1d.restype = self.tensor_handle

        if hasattr(self.lib, "tensor_batchnorm2d"):
            self.lib.tensor_batchnorm2d.argtypes = [
                self.tensor_handle,
                self.tensor_handle,
                self.tensor_handle,
                self.tensor_handle,
                self.tensor_handle,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
            ]
            self.lib.tensor_batchnorm2d.restype = self.tensor_handle

        self.lib.tensor_conv2d.argtypes = [
            self.tensor_handle,
            self.tensor_handle,
            self.tensor_handle,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self.lib.tensor_conv2d.restype = self.tensor_handle

        self.lib.tensor_maxpool2d.argtypes = [self.tensor_handle, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)]
        self.lib.tensor_maxpool2d.restype = self.tensor_handle

        self.lib.tensor_mse_loss.argtypes = [self.tensor_handle, self.tensor_handle]
        self.lib.tensor_mse_loss.restype = self.tensor_handle

        if hasattr(self.lib, "tensor_softmax2d"):
            self.lib.tensor_softmax2d.argtypes = [self.tensor_handle]
            self.lib.tensor_softmax2d.restype = self.tensor_handle

        if hasattr(self.lib, "tensor_categorical_crossentropy"):
            self.lib.tensor_categorical_crossentropy.argtypes = [
                self.tensor_handle,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int64,
                ctypes.c_double,
            ]
            self.lib.tensor_categorical_crossentropy.restype = self.tensor_handle

        if hasattr(self.lib, "tensor_categorical_crossentropy_logits"):
            self.lib.tensor_categorical_crossentropy_logits.argtypes = [
                self.tensor_handle,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int64,
            ]
            self.lib.tensor_categorical_crossentropy_logits.restype = self.tensor_handle

    def last_error(self) -> str:
        msg = self.lib.tensor_last_error()
        if not msg:
            return ""
        return msg.decode("utf-8", errors="replace")


_lib_singleton: Optional[TensorLib] = None


def lib(path: Optional[str] = None) -> TensorLib:
    global _lib_singleton
    if _lib_singleton is None or path is not None:
        _lib_singleton = TensorLib(path)
    return _lib_singleton


class RMSprop:
    """RMSprop optimizer update implemented in `librmsprop.dylib`.

    This calls the C++ function `rmsprop_update(param, velocity, lr, rho, eps, weight_decay)`.
    """

    def __init__(self, path: Optional[str] = None):
        # Ensure libtensor is loaded first (rmsprop dylib depends on tensor_* symbols).
        _ = lib()

        if path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(here, "librmsprop.dylib")
        self.path = path

        mode = getattr(ctypes, "RTLD_GLOBAL", None)
        if mode is None:
            self._lib = ctypes.CDLL(self.path)
        else:
            self._lib = ctypes.CDLL(self.path, mode=mode)

        self._lib.rmsprop_update.argtypes = [
            lib().tensor_handle,
            lib().tensor_handle,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        self._lib.rmsprop_update.restype = ctypes.c_int

    def step(
        self,
        param: "Tensor",
        velocity: "Tensor",
        *,
        lr: float,
        rho: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        rc = int(
            self._lib.rmsprop_update(
                param.handle,
                velocity.handle,
                float(lr),
                float(rho),
                float(eps),
                float(weight_decay),
            )
        )
        if rc != 0:
            raise RuntimeError(lib().last_error() or "rmsprop_update failed")


class Adam:
    """Adam optimizer update implemented in `libadam.dylib`.

    This calls the C++ function:
      adam_update(param, m, v, lr, beta1, beta2, eps, weight_decay, t)
    """

    def __init__(self, path: Optional[str] = None):
        # Ensure libtensor is loaded first (adam dylib depends on tensor_* symbols).
        _ = lib()

        if path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(here, "libadam.dylib")
        self.path = path

        mode = getattr(ctypes, "RTLD_GLOBAL", None)
        if mode is None:
            self._lib = ctypes.CDLL(self.path)
        else:
            self._lib = ctypes.CDLL(self.path, mode=mode)

        self._lib.adam_update.argtypes = [
            lib().tensor_handle,
            lib().tensor_handle,
            lib().tensor_handle,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int64,
        ]
        self._lib.adam_update.restype = ctypes.c_int

    def step(
        self,
        param: "Tensor",
        m: "Tensor",
        v: "Tensor",
        *,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        t: int,
    ) -> None:
        rc = int(
            self._lib.adam_update(
                param.handle,
                m.handle,
                v.handle,
                float(lr),
                float(beta1),
                float(beta2),
                float(eps),
                float(weight_decay),
                int(t),
            )
        )
        if rc != 0:
            raise RuntimeError(lib().last_error() or "adam_update failed")


def _as_int64_array(vals: Sequence[int]) -> Tuple[ctypes.Array, int]:
    arr = (ctypes.c_int64 * len(vals))(*[int(v) for v in vals])
    return arr, len(vals)


def _as_double_array(vals: Sequence[float]) -> Tuple[ctypes.Array, int]:
    arr = (ctypes.c_double * len(vals))(*[float(v) for v in vals])
    return arr, len(vals)


class Tensor:
    def __init__(self, handle: ctypes.c_void_p):
        if not handle:
            raise RuntimeError(lib().last_error() or "null tensor handle")
        self._h = ctypes.c_void_p(handle)

    @property
    def handle(self) -> ctypes.c_void_p:
        return self._h

    def free(self) -> None:
        if getattr(self, "_h", None):
            lib().lib.tensor_free(self._h)
            self._h = ctypes.c_void_p(None)

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    @staticmethod
    def create(data: Sequence[float], shape: Sequence[int], requires_grad: bool = False) -> "Tensor":
        d_arr, d_len = _as_double_array(list(data))
        s_arr, s_len = _as_int64_array(list(shape))
        h = lib().lib.tensor_create(d_arr, d_len, s_arr, s_len, 1 if requires_grad else 0)
        return Tensor(h)

    @staticmethod
    def zeros(shape: Sequence[int], requires_grad: bool = False) -> "Tensor":
        s_arr, s_len = _as_int64_array(list(shape))
        h = lib().lib.tensor_zeros(s_arr, s_len, 1 if requires_grad else 0)
        return Tensor(h)

    @staticmethod
    def ones(shape: Sequence[int], requires_grad: bool = False) -> "Tensor":
        s_arr, s_len = _as_int64_array(list(shape))
        h = lib().lib.tensor_ones(s_arr, s_len, 1 if requires_grad else 0)
        return Tensor(h)

    @staticmethod
    def randn(
        shape: Sequence[int],
        mean: float = 0.0,
        stdev: float = 1.0,
        requires_grad: bool = False,
        seed: Optional[int] = None,
    ) -> "Tensor":
        s_arr, s_len = _as_int64_array(list(shape))
        seed_ptr = None
        if seed is not None:
            seed_val = ctypes.c_int64(int(seed))
            seed_ptr = ctypes.pointer(seed_val)
        h = lib().lib.tensor_randn(s_arr, s_len, float(mean), float(stdev), 1 if requires_grad else 0, seed_ptr)
        return Tensor(h)

    def ndim(self) -> int:
        v = int(lib().lib.tensor_ndim(self._h))
        if v < 0:
            raise RuntimeError(lib().last_error())
        return v

    def size(self) -> int:
        v = int(lib().lib.tensor_size(self._h))
        if v < 0:
            raise RuntimeError(lib().last_error())
        return v

    def shape(self) -> Tuple[int, ...]:
        n = self.ndim()
        out = (ctypes.c_int64 * n)()
        rc = lib().lib.tensor_shape(self._h, out, n)
        if rc != 0:
            raise RuntimeError(lib().last_error())
        return tuple(int(out[i]) for i in range(n))

    def to_list(self) -> list:
        n = self.size()
        out = (ctypes.c_double * n)()
        rc = lib().lib.tensor_read_data(self._h, out, n)
        if rc != 0:
            raise RuntimeError(lib().last_error())
        return [float(out[i]) for i in range(n)]

    def write_from_list(self, data: Sequence[float]) -> None:
        if not hasattr(lib().lib, "tensor_write_data"):
            raise RuntimeError("tensor_write_data not available (rebuild libtensor.dylib)")
        n = self.size()
        vals = list(float(x) for x in data)
        if len(vals) != n:
            raise ValueError("data length must equal tensor.size()")
        arr = (ctypes.c_double * n)(*vals)
        rc = lib().lib.tensor_write_data(self._h, arr, n)
        if rc != 0:
            raise RuntimeError(lib().last_error())

    def has_grad(self) -> bool:
        if not hasattr(lib().lib, "tensor_has_grad"):
            return False
        return bool(int(lib().lib.tensor_has_grad(self._h)))

    def grad_to_list(self) -> list:
        n = self.size()
        out = (ctypes.c_double * n)()
        rc = lib().lib.tensor_read_grad(self._h, out, n)
        if rc != 0:
            raise RuntimeError(lib().last_error())
        return [float(out[i]) for i in range(n)]

    def zero_grad(self) -> None:
        lib().lib.tensor_zero_grad(self._h)

    def backward(self) -> None:
        rc = lib().lib.tensor_backward(self._h)
        if rc != 0:
            raise RuntimeError(lib().last_error())

    def backward_scalar(self, grad: float) -> None:
        rc = lib().lib.tensor_backward_scalar(self._h, float(grad))
        if rc != 0:
            raise RuntimeError(lib().last_error())

    def backward_with_tensor(self, grad: "Tensor") -> None:
        rc = lib().lib.tensor_backward_with_tensor(self._h, grad._h)
        if rc != 0:
            raise RuntimeError(lib().last_error())

    def detach(self) -> "Tensor":
        h = lib().lib.tensor_detach(self._h)
        return Tensor(h)

    def reshape(self, new_shape: Sequence[int]) -> "Tensor":
        s_arr, s_len = _as_int64_array(list(new_shape))
        h = lib().lib.tensor_reshape(self._h, s_arr, s_len)
        return Tensor(h)

    def transpose(self) -> "Tensor":
        h = lib().lib.tensor_transpose(self._h)
        return Tensor(h)

    def sum(self) -> "Tensor":
        h = lib().lib.tensor_sum(self._h)
        return Tensor(h)

    def neg(self) -> "Tensor":
        h = lib().lib.tensor_neg(self._h)
        return Tensor(h)

    def add(self, other: "Tensor") -> "Tensor":
        h = lib().lib.tensor_add(self._h, other._h)
        return Tensor(h)

    def add_scalar(self, c: float) -> "Tensor":
        h = lib().lib.tensor_add_scalar(self._h, float(c))
        return Tensor(h)

    def sub(self, other: "Tensor") -> "Tensor":
        h = lib().lib.tensor_sub(self._h, other._h)
        return Tensor(h)

    def sub_scalar(self, c: float) -> "Tensor":
        h = lib().lib.tensor_sub_scalar(self._h, float(c))
        return Tensor(h)

    def mul(self, other: "Tensor") -> "Tensor":
        h = lib().lib.tensor_mul(self._h, other._h)
        return Tensor(h)

    def mul_scalar(self, c: float) -> "Tensor":
        h = lib().lib.tensor_mul_scalar(self._h, float(c))
        return Tensor(h)

    def div(self, other: "Tensor") -> "Tensor":
        h = lib().lib.tensor_div(self._h, other._h)
        return Tensor(h)

    def div_scalar(self, c: float) -> "Tensor":
        h = lib().lib.tensor_div_scalar(self._h, float(c))
        return Tensor(h)

    def pow(self, p: float) -> "Tensor":
        h = lib().lib.tensor_pow(self._h, float(p))
        return Tensor(h)

    def matmul(self, other: "Tensor") -> "Tensor":
        h = lib().lib.tensor_matmul(self._h, other._h)
        return Tensor(h)

    def relu(self) -> "Tensor":
        h = lib().lib.tensor_relu(self._h)
        return Tensor(h)

    def dropout(self, p: float = 0.5, seed: Optional[int] = None, training: bool = True) -> "Tensor":
        if not hasattr(lib().lib, "tensor_dropout"):
            raise RuntimeError("tensor_dropout not available (rebuild libtensor.dylib)")
        seed_ptr = None
        if seed is not None:
            seed_val = ctypes.c_int64(int(seed))
            seed_ptr = ctypes.pointer(seed_val)
        h = lib().lib.tensor_dropout(self._h, float(p), seed_ptr, 1 if training else 0)
        return Tensor(h)

    def batchnorm1d(
        self,
        gamma: "Tensor",
        beta: "Tensor",
        running_mean: Optional["Tensor"] = None,
        running_var: Optional["Tensor"] = None,
        momentum: float = 0.1,
        eps: float = 1e-5,
        training: bool = True,
    ) -> "Tensor":
        if not hasattr(lib().lib, "tensor_batchnorm1d"):
            raise RuntimeError("tensor_batchnorm1d not available (rebuild libtensor.dylib)")
        rm = running_mean._h if running_mean is not None else ctypes.c_void_p(None)
        rv = running_var._h if running_var is not None else ctypes.c_void_p(None)
        h = lib().lib.tensor_batchnorm1d(
            self._h,
            gamma._h,
            beta._h,
            rm,
            rv,
            float(momentum),
            float(eps),
            1 if training else 0,
        )
        return Tensor(h)

    def batchnorm2d(
        self,
        gamma: "Tensor",
        beta: "Tensor",
        running_mean: Optional["Tensor"] = None,
        running_var: Optional["Tensor"] = None,
        momentum: float = 0.1,
        eps: float = 1e-5,
        training: bool = True,
    ) -> "Tensor":
        if not hasattr(lib().lib, "tensor_batchnorm2d"):
            raise RuntimeError("tensor_batchnorm2d not available (rebuild libtensor.dylib)")
        rm = running_mean._h if running_mean is not None else ctypes.c_void_p(None)
        rv = running_var._h if running_var is not None else ctypes.c_void_p(None)
        h = lib().lib.tensor_batchnorm2d(
            self._h,
            gamma._h,
            beta._h,
            rm,
            rv,
            float(momentum),
            float(eps),
            1 if training else 0,
        )
        return Tensor(h)

    def conv2d(
        self,
        w: "Tensor",
        b: Optional["Tensor"] = None,
        stride: int = 1,
        padding: int = 0,
    ) -> "Tensor":
        bh = b._h if b is not None else ctypes.c_void_p(None)
        h = lib().lib.tensor_conv2d(self._h, w._h, bh, int(stride), int(padding))
        return Tensor(h)

    def maxpool2d(self, kernel: int = 2, stride: Optional[int] = None) -> "Tensor":
        stride_ptr = None
        if stride is not None:
            stride_val = ctypes.c_int64(int(stride))
            stride_ptr = ctypes.pointer(stride_val)
        h = lib().lib.tensor_maxpool2d(self._h, int(kernel), stride_ptr)
        return Tensor(h)

    def mse_loss(self, target: "Tensor") -> "Tensor":
        h = lib().lib.tensor_mse_loss(self._h, target._h)
        return Tensor(h)

    def softmax2d(self) -> "Tensor":
        if not hasattr(lib().lib, "tensor_softmax2d"):
            raise RuntimeError("tensor_softmax2d not available (rebuild libtensor.dylib)")
        h = lib().lib.tensor_softmax2d(self._h)
        return Tensor(h)

    def categorical_crossentropy(self, targets: Sequence[int], eps: float = 1e-12) -> "Tensor":
        if not hasattr(lib().lib, "tensor_categorical_crossentropy"):
            raise RuntimeError("tensor_categorical_crossentropy not available (rebuild libtensor.dylib)")
        t_arr, t_len = _as_int64_array(list(targets))
        h = lib().lib.tensor_categorical_crossentropy(self._h, t_arr, t_len, float(eps))
        return Tensor(h)

    def categorical_crossentropy_logits(self, targets: Sequence[int]) -> "Tensor":
        if not hasattr(lib().lib, "tensor_categorical_crossentropy_logits"):
            raise RuntimeError("tensor_categorical_crossentropy_logits not available (rebuild libtensor.dylib)")
        t_arr, t_len = _as_int64_array(list(targets))
        h = lib().lib.tensor_categorical_crossentropy_logits(self._h, t_arr, t_len)
        return Tensor(h)

