/*
Copyright (c) 2018 Uber Technologies, Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <string>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tf_env.h"
//#include "indexedmatmul.h"

#ifdef __USE_SDL
  #include <SDL.h>
#endif

using namespace tensorflow;
using namespace std;

EnvironmentMakeOp::EnvironmentMakeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                    context->allocate_persistent(DT_STRING, TensorShape({2}),
                                                &handle_, nullptr));

    OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size));
}

// The resource is deleted from the resource manager only when it is private
// to kernel. Ideally the resource should be deleted when it is no longer held
// by anyone, but it would break backward compatibility.
EnvironmentMakeOp::~EnvironmentMakeOp() {
    if (resource_ != nullptr) {
        while (!resource_->RefCountIsOne())
        {
            resource_->Unref();
            cout << "***Failed to unreference resources***\n";
        }
        resource_->Unref();
        if (cinfo_.resource_is_private_to_kernel()) {
            if (!cinfo_.resource_manager()
                ->template Delete<BaseEnvironment>(cinfo_.container(), cinfo_.name())
                                                    .ok()) {
            // Do nothing; the resource can have been deleted by session resets.
            }
        }
    }
}

void EnvironmentMakeOp::Compute(OpKernelContext* context) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    if (resource_ == nullptr) {
        ResourceMgr* mgr = context->resource_manager();
        OP_REQUIRES_OK(context, cinfo_.Init(mgr, def()));

        BaseEnvironment* resource;
        OP_REQUIRES_OK(
            context,
            mgr->LookupOrCreate<BaseEnvironment>(cinfo_.container(), cinfo_.name(), &resource,
                                    [context, this](BaseEnvironment** ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                                    Status s = CreateResource(context, ret);
                                    if (!s.ok() && *ret != nullptr) {
                                        CHECK((*ret)->Unref());
                                    }
                                    return s;
                                    }));

        Status s = VerifyResource(resource);
        if (TF_PREDICT_FALSE(!s.ok())) {
        resource->Unref();
        context->SetStatus(s);
        return;
        }

        auto h = handle_.AccessTensor(context)->template flat<string>();
        h(0) = cinfo_.container();
        h(1) = cinfo_.name();
        resource_ = resource;
    }
    if (context->expected_output_dtype(0) == DT_RESOURCE) {
        OP_REQUIRES_OK(context, MakeResourceHandleToOutput(
                                    context, 0, cinfo_.container(), cinfo_.name(),
                                    MakeTypeIndex<BaseEnvironment>()));
    } else {
        context->set_output_ref(0, &mu_, handle_.AccessTensor(context));
    }
}

// During the first Compute(), resource is either created or looked up using
// shared_name. In the latter case, the resource found should be verified if
// it is compatible with this op's configuration. The verification may fail in
// cases such as two graphs asking queues of the same shared name to have
// inconsistent capacities.
Status EnvironmentMakeOp::VerifyResource(BaseEnvironment* resource) { return Status::OK(); }

class EnvironmentResetOp : public OpKernel {
    public:
    explicit EnvironmentResetOp(OpKernelConstruction* context) : OpKernel(context) {
    }
    void Compute(OpKernelContext* context) override {
        const Tensor& indices_tensor = context->input(1);
        auto indices_flat = indices_tensor.flat<int>();

        const Tensor& noops_tensor = context->input(2);
        auto noops_flat = noops_tensor.flat<int>();

        const Tensor& max_frames_tensor = context->input(3);
        auto max_frames_flat = max_frames_tensor.flat<int>();

        const int m_numInterfaces = indices_tensor.NumElements();

        BaseEnvironment *env;
        OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0), &env));
        core::ScopedUnref s(env);

        const auto thread_pool = context->device()->tensorflow_cpu_worker_threads();
        const int num_threads = std::min(thread_pool->num_threads, int(m_numInterfaces));
        auto f = [&](int thread_id) {
            // Set all but the first element of the output tensor to 0.
            for(int b =thread_id; b < m_numInterfaces;b+=num_threads)
            {
                env->reset(indices_flat(b), noops_flat(b), max_frames_flat(b));
            }
        };

        BlockingCounter counter(num_threads-1);
        for (int i = 1; i < num_threads; ++i) {
            thread_pool->workers->Schedule([&, i]() {
                f(i);
                counter.DecrementCount();
            });
        }
        f(0);
        counter.Wait();
    }
};

REGISTER_OP("EnvironmentReset")
    .Input("handle: resource")
    .Input("indices: int32")
    .Input("noops: int32")
    .Input("max_frames: int32")
    .SetShapeFn(::tensorflow::shape_inference::UnknownShape);

REGISTER_KERNEL_BUILDER(Name("EnvironmentReset").Device(DEVICE_CPU), EnvironmentResetOp);



template<typename T>
class EnvironmentObservationOp : public OpKernel {
    public:
    explicit EnvironmentObservationOp(OpKernelConstruction* context) : OpKernel(context) {
    }
    void Compute(OpKernelContext* context) override {
        const Tensor& indices_tensor = context->input(1);
        auto indices_flat = indices_tensor.flat<int>();
        const int m_numInterfaces = indices_tensor.NumElements();


        BaseEnvironment *tmp_env;
        OP_REQUIRES_OK(context, LookupResource<BaseEnvironment>(context, HandleFromInput(context, 0), &tmp_env));
        Environment<T> *env = dynamic_cast<Environment<T>*>(tmp_env);
        core::ScopedUnref s(env);

        Tensor* image_tensor = NULL;
        TensorShape shape = env->get_observation_shape();
        const size_t ssize = shape.num_elements();
        shape.InsertDim(0, m_numInterfaces);
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &image_tensor));
        auto image_flat = image_tensor->flat<T>();

        const auto thread_pool = context->device()->tensorflow_cpu_worker_threads();
        const int num_threads = std::min(thread_pool->num_threads, int(m_numInterfaces));

        auto f = [&](int thread_id) {
            // Set all but the first element of the output tensor to 0.
            for(int b =thread_id; b < m_numInterfaces;b+=num_threads)
            {
                env->get_observation(&image_flat(b * ssize), indices_flat(b));
            }
        };

        BlockingCounter counter(num_threads-1);
        for (int i = 1; i < num_threads; ++i) {
            thread_pool->workers->Schedule([&, i]() {
                f(i);
                counter.DecrementCount();
            });
        }
        f(0);
        counter.Wait();
    }
};

REGISTER_OP("EnvironmentObservation")
    .Attr("T: {uint8, float}")
    .Input("handle: resource")
    .Input("indices: int32")
    .Output("image: T")
    .SetShapeFn(::tensorflow::shape_inference::UnknownShape);

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("EnvironmentObservation").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      EnvironmentObservationOp<type>)
REGISTER_KERNEL(uint8)
REGISTER_KERNEL(float)
#undef REGISTER_KERNEL


template<typename T>
class EnvironmentStepOp : public OpKernel {
    public:
    explicit EnvironmentStepOp(OpKernelConstruction* context) : OpKernel(context) {
    }
    void Compute(OpKernelContext* context) override {
        const Tensor &indices_tensor = context->input(1);
        auto indices_flat = indices_tensor.flat<int>();

        const Tensor& action_tensor = context->input(2);
        auto action_flat = action_tensor.flat<T>();

        const int m_numInterfaces = indices_tensor.NumElements();

        Tensor* reward_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,
                                            TensorShape({static_cast<int>(m_numInterfaces)}),
                                            &reward_tensor));
        auto reward_flat = reward_tensor->flat<float>();

        Tensor* done_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1,
                                            TensorShape({static_cast<int>(m_numInterfaces)}),
                                            &done_tensor));
        auto done_flat = done_tensor->flat<bool>();

        BaseEnvironment *tmp_env;
        OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0), &tmp_env));

        auto env = dynamic_cast<StepInterface<T> *>(tmp_env);
        OP_REQUIRES(context, env != nullptr, errors::Internal("BaseEnvironment is not of StepInterface<T> type."));
        core::ScopedUnref s(tmp_env);

        for(int b = 0; b < m_numInterfaces;++b)
            OP_REQUIRES(context, !tmp_env->is_done(indices_flat(b)), errors::Internal("BaseEnvironment episode already completed."));

        const auto ssize = env->get_action_shape().num_elements();
        const auto thread_pool = context->device()->tensorflow_cpu_worker_threads();
        const int num_threads = std::min(thread_pool->num_threads, int(m_numInterfaces));

        auto f = [&](int thread_id) {
            // Set all but the first element of the output tensor to 0.
            for(int b =thread_id; b < m_numInterfaces;b+=num_threads)
            {
                reward_flat(b) = env->step(indices_flat(b), &action_flat(b * ssize));
                done_flat(b) = tmp_env->is_done(indices_flat(b));
            }
        };

        BlockingCounter counter(num_threads-1);
        for (int i = 1; i < num_threads; ++i) {
            thread_pool->workers->Schedule([&, i]() {
                f(i);
                counter.DecrementCount();
            });
        }
        f(0);
        counter.Wait();
    }
};

REGISTER_OP("EnvironmentStep")
    .Attr("T: {int32, float}")
    .Input("handle: resource")
    .Input("indices: int32")
    .Input("action: T")
    .Output("reward: float")
    .Output("done: bool")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        ::tensorflow::shape_inference::ShapeHandle input;
        ::tensorflow::shape_inference::ShapeHandle output;
        for (size_t i = 1; i < c->num_inputs()-1; ++i) {
            TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &input));
            TF_RETURN_IF_ERROR(c->Merge(output, input, &output));
        }
        c->set_output(0, output);
        c->set_output(1, output);
        return Status::OK();
    });


#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("EnvironmentStep").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      EnvironmentStepOp<type>)
REGISTER_KERNEL(int)
REGISTER_KERNEL(float)

#undef REGISTER_KERNEL
