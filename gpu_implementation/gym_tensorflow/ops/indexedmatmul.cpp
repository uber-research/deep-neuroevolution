#define EIGEN_USE_THREADS

#include <vector>
#include <iostream>

using namespace std;
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

#include "tensorflow/core/framework/shape_inference.h"
#ifdef GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif
typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace {
#ifdef GOOGLE_CUDA
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}

class CublasScratchAllocator : public perftools::gputools::ScratchAllocator {
 public:
  using Stream = ::perftools::gputools::Stream;
  using DeviceMemoryBytes = ::perftools::gputools::DeviceMemory<uint8>;

  CublasScratchAllocator(OpKernelContext* context) : context_(context) {}

  int64 GetMemoryLimitInBytes(Stream* stream) override { return -1; }

  perftools::gputools::port::StatusOr<DeviceMemoryBytes> AllocateBytes(
      Stream* stream, int64 byte_size) override {
    Tensor temporary_memory;

    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory));
    if (!allocation_status.ok()) {
      return perftools::gputools::port::StatusOr<DeviceMemoryBytes>(
          DeviceMemoryBytes::MakeFromByteSize(nullptr, 0));
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    return perftools::gputools::port::StatusOr<DeviceMemoryBytes>(
        DeviceMemoryBytes::MakeFromByteSize(
            temporary_memory.flat<uint8>().data(),
            temporary_memory.flat<uint8>().size()));
  }

 private:
  OpKernelContext* context_;
  std::vector<Tensor> allocated_tensors_;
};
#endif

// TODO(rmlarsen): Get rid of this when we have upstreamed improvements
// for matrix*vector and vector*matrix to Eigen's general matrix product.
template <typename Tx, typename Ty, typename Tz>
static void Multiply(bool adj_x, bool adj_y, Tx x, Ty y, Tz z) {
  if (!adj_x) {
    if (!adj_y) {
      z.noalias() = x * y;
    } else {
      z.noalias() = x * y.adjoint();
    }
  } else {
    if (!adj_y) {
      z.noalias() = x.adjoint() * y;
    } else {
      z.noalias() = x.adjoint() * y.adjoint();
    }
  }
}

// Sequential batch matmul kernel that calls the regular Eigen matmul.
// We prefer this over the tensor contraction because it performs
// better on vector-matrix and matrix-vector products.
template <typename Scalar>
struct SequentialIndexedMatMulKernel {
  using Matrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

  static ConstMatrixMap ConstTensorSliceToEigenMatrix(const Tensor& t,
                                                      int slice) {
    return ConstMatrixMap(
        t.flat<Scalar>().data() + slice * t.dim_size(1) * t.dim_size(2),
        t.dim_size(1), t.dim_size(2));
  }

  static MatrixMap TensorSliceToEigenMatrix(Tensor* t, int slice) {
    return MatrixMap(
        t->flat<Scalar>().data() + slice * t->dim_size(1) * t->dim_size(2),
        t->dim_size(1), t->dim_size(2));
  }

  static void Run(const Tensor& in_x, const Tensor& in_y, const Tensor& in_idx, bool adj_x,
                  bool adj_y, Tensor* out, int start, int limit) {
    auto idx = in_idx.flat<int>();
    for (int i = start; i < limit; ++i) {
      auto x = ConstTensorSliceToEigenMatrix(in_x, i);
      auto y = ConstTensorSliceToEigenMatrix(in_y, idx(i));
      auto z = TensorSliceToEigenMatrix(out, i);
      // TODO(rmlarsen): Get rid of the special casing here when we have
      // upstreamed improvements for matrix*vector and vector*matrix to
      // Eigen's general matrix product.
      if (!adj_x && x.rows() == 1) {
        Multiply(adj_x, adj_y, x.row(0), y, z);
      } else if (adj_x && x.cols() == 1) {
        Multiply(adj_x, adj_y, x.col(0), y, z);
      } else if (!adj_y && y.cols() == 1) {
        Multiply(adj_x, adj_y, x, y.col(0), z);
      } else if (adj_y && y.rows() == 1) {
        Multiply(adj_x, adj_y, x, y.row(0), z);
      } else {
        Multiply(adj_x, adj_y, x, y, z);
      }
    }
  }
};

}



template <typename Device, typename Scalar>
struct LaunchIndexedBatchMatMul;

#ifdef GOOGLE_CUDA

template <typename Scalar>
struct LaunchIndexedBatchMatMul<GPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, const Tensor& in_idx, bool adj_x, bool adj_y, Tensor* out) {
    constexpr perftools::gputools::blas::Transpose kTranspose =
        is_complex<Scalar>::value
            ? perftools::gputools::blas::Transpose::kConjugateTranspose
            : perftools::gputools::blas::Transpose::kTranspose;
    perftools::gputools::blas::Transpose trans[] = {
        perftools::gputools::blas::Transpose::kNoTranspose, kTranspose};
    const uint64 m = in_x.dim_size(adj_x ? 2 : 1);
    const uint64 k = in_x.dim_size(adj_x ? 1 : 2);
    const uint64 n = in_y.dim_size(adj_y ? 1 : 2);
    const uint64 batch_size = in_x.dim_size(0);
    auto blas_transpose_a = trans[adj_x];
    auto blas_transpose_b = trans[adj_y];

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    typedef perftools::gputools::DeviceMemory<Scalar> DeviceMemoryType;
    std::vector<DeviceMemoryType> a_device_memory;
    std::vector<DeviceMemoryType> b_device_memory;
    std::vector<DeviceMemoryType> c_device_memory;
    std::vector<DeviceMemoryType*> a_ptrs;
    std::vector<DeviceMemoryType*> b_ptrs;
    std::vector<DeviceMemoryType*> c_ptrs;
    a_device_memory.reserve(batch_size);
    b_device_memory.reserve(batch_size);
    c_device_memory.reserve(batch_size);
    a_ptrs.reserve(batch_size);
    b_ptrs.reserve(batch_size);
    c_ptrs.reserve(batch_size);
    auto* a_base_ptr = in_x.template flat<Scalar>().data();
    auto* b_base_ptr = in_y.template flat<Scalar>().data();
    auto* c_base_ptr = out->template flat<Scalar>().data();

    auto idx = in_idx.flat<int>();
    for (int64 i = 0; i < batch_size; ++i) {
        a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
        b_device_memory.push_back(AsDeviceMemory(b_base_ptr + idx(i) * k * n));
        c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
        a_ptrs.push_back(&a_device_memory.back());
        b_ptrs.push_back(&b_device_memory.back());
        c_ptrs.push_back(&c_device_memory.back());
    }

    CublasScratchAllocator scratch_allocator(context);
    bool blas_launch_status =
        stream
            ->ThenBlasGemmBatchedWithScratch(
                blas_transpose_b, blas_transpose_a, n, m, k,
                static_cast<Scalar>(1.0), b_ptrs, adj_y ? k : n, a_ptrs,
                adj_x ? m : k, static_cast<Scalar>(0.0), c_ptrs, n,
                batch_size, &scratch_allocator)
            .ok();
    if (!blas_launch_status) {
    context->SetStatus(errors::Internal(
        "Blas xGEMMBatched launch failed : a.shape=",
        in_x.shape().DebugString(),
        ", b.shape=", in_y.shape().DebugString(),
        ", idx.shape=", in_idx.shape().DebugString(),
        ", m=", m, ", n=", n,
        ", k=", k, ", batch_size=", batch_size));
    }
  }
};
#endif

template <typename Scalar>
struct LaunchIndexedBatchMatMul<CPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, const Tensor& in_idx, bool adj_x, bool adj_y, Tensor* out) {
    // Number of matrix multiplies i.e. size of the batch.
    const int64 batch_size = in_idx.dim_size(0);
    const int64 cost_per_unit =
        in_x.dim_size(1) * in_x.dim_size(2) * out->dim_size(2);
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
      // Parallelize over outer dims. For small matrices and large batches, it
      // is counter-productive to parallelize the inner matrix multiplies.
      Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
            cost_per_unit,
            [&in_x, &in_y, &in_idx, adj_x, adj_y, out](int start, int limit) {
              SequentialIndexedMatMulKernel<Scalar>::Run(in_x, in_y, in_idx, adj_x, adj_y, out,
                                                  start, limit);
            });
  }
};

template <typename Device, typename Scalar>
class IndexedBatchMatMul : public OpKernel {
 public:
  explicit IndexedBatchMatMul(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));
  }

  virtual ~IndexedBatchMatMul() {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    const Tensor& in2 = ctx->input(2);
    OP_REQUIRES(ctx, in0.dims() == in1.dims(),
                errors::InvalidArgument("In[0] and In[1] has different ndims: ",
                                        in0.shape().DebugString(), " vs. ",
                                        in1.shape().DebugString()));
    const int ndims = in0.dims();
    OP_REQUIRES(
        ctx, ndims >= 2,
        errors::InvalidArgument("In[0] and In[1] ndims must be >= 2: ", ndims));

    TensorShape out_shape;
    for (int i = 0; i < ndims - 2; ++i) {
      OP_REQUIRES(ctx, in0.dim_size(i) == in2.dim_size(i),
                  errors::InvalidArgument(
                      "In[0].dim(", i, ") and In[2].dim(", i,
                      ") must be the same: ", in0.shape().DebugString(), " vs ",
                      in2.shape().DebugString()));
      out_shape.AddDim(in0.dim_size(i));
    }
    auto n = out_shape.num_elements();
    auto d0 = in0.dim_size(ndims - 2);
    auto d1 = in0.dim_size(ndims - 1);
    Tensor in0_reshaped;
    CHECK(in0_reshaped.CopyFrom(in0, TensorShape({n, d0, d1})));
    auto n2 = in1.dim_size(0);
    auto d2 = in1.dim_size(ndims - 2);
    auto d3 = in1.dim_size(ndims - 1);
    Tensor in1_reshaped;
    CHECK(in1_reshaped.CopyFrom(in1, TensorShape({n2, d2, d3})));
    if (adj_x_) std::swap(d0, d1);
    if (adj_y_) std::swap(d2, d3);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument(
                    "In[0] mismatch In[1] shape: ", d1, " vs. ", d2, ": ",
                    in0.shape().DebugString(), " ", in1.shape().DebugString(),
                    " ", adj_x_, " ", adj_y_));
    out_shape.AddDim(d0);
    out_shape.AddDim(d3);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    Tensor out_reshaped;
    CHECK(out_reshaped.CopyFrom(*out, TensorShape({n, d0, d3})));
    LaunchIndexedBatchMatMul<Device, Scalar>::Launch(ctx, in0_reshaped, in1_reshaped, in2,
                                              adj_x_, adj_y_, &out_reshaped);
  }

 private:
  bool adj_x_;
  bool adj_y_;
};

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
REGISTER_OP("IndexedBatchMatMul")
    .Attr("adj_y: bool = false")
    .Attr("adj_x: bool = false")
    .Attr("T: {float}")
    .Input("x: T")
    .Input("y: T")
    .Input("idx: int32")
    .Output("out: T")
    .SetShapeFn([](InferenceContext *c) {
        ShapeHandle a_shape;
        ShapeHandle b_shape;
        ShapeHandle c_shape;
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &a_shape));
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &b_shape));
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 1, &c_shape));

        // Determine output rows and cols.
        bool adj_x;
        bool adj_y;
        TF_RETURN_IF_ERROR(c->GetAttr("adj_x", &adj_x));
        TF_RETURN_IF_ERROR(c->GetAttr("adj_y", &adj_y));
        DimensionHandle output_rows = c->Dim(a_shape, adj_x ? -1 : -2);
        DimensionHandle output_cols = c->Dim(b_shape, adj_y ? -2 : -1);

        // Batch dims match between inputs.
        ShapeHandle a_batch_dims;
        ShapeHandle c_batch_dims = c_shape;
        ShapeHandle batch_dims;
        TF_RETURN_IF_ERROR(c->Subshape(a_shape, 0, -2, &a_batch_dims));
        TF_RETURN_IF_ERROR(c->Merge(a_batch_dims, c_batch_dims, &batch_dims));

        // Assert inner dims match.
        DimensionHandle unused;
        TF_RETURN_IF_ERROR(c->Merge(c->Dim(a_shape, adj_x ? -2 : -1),
                                    c->Dim(b_shape, adj_y ? -1 : -2), &unused));

        ShapeHandle out;
        TF_RETURN_IF_ERROR(c->Concatenate(
            batch_dims, c->Matrix(output_rows, output_cols), &out));
        c->set_output(0, out);
        return Status::OK();
    });

#ifdef GOOGLE_CUDA
#define REGISTER_BATCH_MATMUL_GPU(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("IndexedBatchMatMul").Device(DEVICE_GPU).TypeConstraint<TYPE>("T").HostMemory("idx"), \
      IndexedBatchMatMul<GPUDevice, TYPE>)
TF_CALL_float(REGISTER_BATCH_MATMUL_GPU);
#endif

#define REGISTER_BATCH_MATMUL_CPU(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("IndexedBatchMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      IndexedBatchMatMul<CPUDevice, TYPE>)

TF_CALL_float(REGISTER_BATCH_MATMUL_CPU);

}  // end namespace tensorflow